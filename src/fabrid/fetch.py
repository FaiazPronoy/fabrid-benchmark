import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import shutil
import statistics
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from fabrid import runio

USER_AGENT = "fabrid-fetch"
CHUNK = 1 << 20
GB = 1024**3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_SUFFIXES = {".xlsx", ".xls", ".json", ".parquet", ".txt", ".mat", ".h5", ".npy"}
MASK_WORDS = {"mask", "masks", "label", "labels", "gt", "annotation", "annotations", "annot"}
MASK_WORDS |= {"seg", "segmentation", "groundtruth", "ground", "truth"}
LABEL_WORDS = ("label", "class", "defect", "type")
SEPARATORS = re.compile(r"[_\-\s]+")
MARKER = ".extracted"
REFERENCE_ROWS = 200_000
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError)


def open_url(url: str, headers: dict | None = None, data: bytes | None = None, timeout=60):
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return urllib.request.urlopen(request, timeout=timeout)


def request_json(url: str, payload: dict | None = None, timeout=60, retries: int = 3):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    for attempt in range(1, retries + 1):
        try:
            with open_url(url, headers, data, timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except NETWORK_ERRORS as error:
            if attempt == retries:
                raise SystemExit(f"cannot reach {url}: {error}") from error
            time.sleep(2 * attempt)
    return None


def bare_doi(doi: str) -> str:
    return re.sub(r"\.v\d+$", "", doi.strip().lower())


def resolve(api: str, doi: str, timeout=60) -> dict:
    hits = request_json(f"{api}/articles/search", {"doi": doi}, timeout) or []
    matches = [hit for hit in hits if bare_doi(hit.get("doi", "")) == bare_doi(doi)]
    if not matches:
        raise SystemExit(f"the Figshare API returned no record for DOI {doi}")
    return request_json(f"{api}/articles/{matches[0]['id']}", None, timeout)


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def gib(value: float) -> str:
    return f"{value / GB:.2f} GB"


def check_space(directory: Path, need: float, free=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    have = shutil.disk_usage(directory).free if free is None else free
    if have < need:
        raise SystemExit(
            f"not enough free disk in {directory}: need {gib(need)}, have {gib(have)}. "
            "Free space or point runs_root at a larger disk."
        )


def download(entry: dict, folder: Path, retries: int, wait: float, timeout=60) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / entry["name"]
    part = target.with_name(target.name + ".part")
    expected, size = entry["md5"], int(entry["size"])
    if target.is_file():
        if target.stat().st_size == size and md5_of(target) == expected:
            return "present"
        target.unlink()
    resumed = part.is_file() and 0 < part.stat().st_size < size
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.is_file() else 0
        if offset > size:
            part.unlink()
            offset = 0
        if offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with open_url(entry["url"], headers, None, timeout) as response:
                append = bool(offset) and response.status == 206
                resumed = resumed or append
                with part.open("ab" if append else "wb") as handle:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        handle.write(block)
        except NETWORK_ERRORS as error:
            got = part.stat().st_size if part.is_file() else 0
            print(f"{entry['name']}: attempt {attempt} stopped at {gib(got)}: {error}")
            time.sleep(wait * attempt)
            continue
        got = part.stat().st_size
        if got < size:
            print(f"{entry['name']}: attempt {attempt} ended early at {gib(got)}")
            time.sleep(wait * attempt)
    if not part.is_file() or part.stat().st_size != size:
        raise SystemExit(f"{entry['name']}: download incomplete after {retries} attempts")
    actual = md5_of(part)
    if actual != expected:
        part.unlink()
        raise SystemExit(
            f"{entry['name']}: md5 {actual} does not match the published {expected}; "
            "the file was removed"
        )
    part.replace(target)
    return "resumed" if resumed else "downloaded"


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def archive_stem(path: Path) -> str:
    return re.sub(r"(\.tar\.gz|\.tgz|\.tar|\.zip)$", "", path.name, flags=re.IGNORECASE)


def unpacked_size(path: Path) -> int:
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            return sum(info.file_size for info in archive.infolist())
    with tarfile.open(path) as archive:
        return sum(member.size for member in archive.getmembers() if member.isfile())


def inside(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if dest.resolve() not in (target, *target.parents):
        raise SystemExit(f"archive member escapes its folder: {name}")
    return target


def unpack(path: Path, dest: Path) -> None:
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = inside(dest, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK)
        return
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            target = inside(dest, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK)


def extract(path: Path, dest: Path, margin: float = 0, free=None) -> str:
    key = f"{path.name} {path.stat().st_size}"
    marker = dest / MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8") == key:
        return "present"
    redone = dest.exists()
    if redone:
        shutil.rmtree(dest)
    check_space(dest.parent, unpacked_size(path) + margin, free)
    dest.mkdir(parents=True)
    unpack(path, dest)
    marker.write_text(key, encoding="utf-8")
    return "redone" if redone else "extracted"


def extract_all(archives, root: Path, depth: int, margin: float) -> tuple[list, list]:
    done, skipped = [], []
    queue = [(path, root / archive_stem(path), 1) for path in archives]
    while queue:
        path, dest, level = queue.pop(0)
        status = extract(path, dest, margin)
        done.append((path, dest, status))
        print(f"{path.name}: {status} into {dest}")
        for nested in sorted(p for p in dest.rglob("*") if p.is_file() and is_archive(p)):
            if level < depth:
                queue.append((nested, nested.with_name(archive_stem(nested)), level + 1))
            else:
                skipped.append(nested)
    return done, skipped


def relative(path: Path, root: Path) -> str:
    text = path.relative_to(root).as_posix()
    return text if text else "."


def walk(root: Path) -> dict[Path, list[Path]]:
    folders = {}
    for current, dirs, files in os.walk(root):
        dirs.sort()
        folders[Path(current)] = sorted(Path(current) / f for f in files if f != MARKER)
    return folders


def tree_rows(folders, root: Path) -> list[list]:
    rows = []
    for folder, files in folders.items():
        extensions = Counter(f.suffix.lower().lstrip(".") or "none" for f in files)
        listed = ";".join(f"{ext}:{n}" for ext, n in sorted(extensions.items()))
        rows.append(
            [relative(folder, root), len(files), listed, sum(f.stat().st_size for f in files)]
        )
    return rows


def pixel_values(array: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    if array.ndim == 2:
        values, counts = np.unique(array, return_counts=True)
        return [str(v.item()) for v in values], counts, array != 0
    flat = array.reshape(-1, array.shape[-1])
    values, counts = np.unique(flat, axis=0, return_counts=True)
    labels = ["(" + ",".join(str(int(c)) for c in v) + ")" for v in values]
    return labels, counts, (array != 0).any(axis=-1)


def zero_label(label: str) -> bool:
    return label == "0" or re.fullmatch(r"\((0,)*0\)", label) is not None


def read_image(path: Path):
    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except Exception:
        return None


def header_of(path: Path):
    try:
        with Image.open(path) as image:
            return image.width, image.height, image.mode
    except Exception:
        return None


def mask_reason(folder: Path, files: list[Path], config: dict) -> str | None:
    named = bool({t for part in folder.parts for t in SEPARATORS.split(part.lower())} & MASK_WORDS)
    sample = files[: config["mask_sample"]]
    few, single = True, True
    for path in sample:
        image = read_image(path)
        if image is None:
            continue
        array = np.asarray(image)
        labels, _, _ = pixel_values(array)
        few &= len(labels) <= config["mask_max_values"]
        single &= array.ndim == 2
    if few:
        return "content and name" if named else "content"
    if named and single:
        return "name"
    return None


def components(foreground: np.ndarray) -> int:
    from scipy import ndimage

    _, count = ndimage.label(foreground, structure=np.ones((3, 3)))
    return int(count)


def summarize_masks(files: list[Path]) -> dict:
    totals = Counter()
    per_mask, sizes, unreadable = [], Counter(), 0
    for path in files:
        image = read_image(path)
        if image is None:
            unreadable += 1
            continue
        array = np.asarray(image)
        labels, counts, foreground = pixel_values(array)
        totals.update(dict(zip(labels, counts.tolist(), strict=True)))
        nonzero = [label for label in labels if not zero_label(label)]
        per_mask.append((len(nonzero), components(foreground), path))
        sizes[f"{image.width}x{image.height}x{image.mode}"] += 1
    nonempty = [m for m in per_mask if m[0] > 0]
    pieces = [m[1] for m in nonempty]
    multi = [m for m in nonempty if m[1] > 1]
    return {
        "masks": len(files),
        "unreadable": unreadable,
        "sizes": sizes,
        "values": totals,
        "all_zero_share": share(len(per_mask) - len(nonempty), len(per_mask)),
        "multi_value_share": share(sum(m[0] > 1 for m in nonempty), len(nonempty)),
        "components_min": min(pieces, default=0),
        "components_median": statistics.median(pieces) if pieces else 0,
        "components_mean": round(statistics.fmean(pieces), 3) if pieces else 0,
        "components_max": max(pieces, default=0),
        "multi_component_share": share(len(multi), len(nonempty)),
        "values_equal_components_share": share(sum(m[0] == m[1] for m in multi), len(multi)),
    }


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def normalized(stem: str) -> str:
    tokens = [t for t in SEPARATORS.split(stem.lower()) if t and t not in MASK_WORDS]
    return "_".join(tokens)


def common_depth(first: Path, second: Path) -> int:
    depth = 0
    for a, b in zip(first.parts, second.parts, strict=False):
        if a != b:
            break
        depth += 1
    return depth


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    return (rows[0], rows[1:]) if rows else ([], [])


def label_tables(folders, root: Path, image_names: set[str], image_stems: set[str]):
    rows, columns, references = [], {}, {}
    for folder, files in folders.items():
        for path in (f for f in files if f.suffix.lower() == ".csv"):
            header, body = read_csv(path)
            name = relative(path, root)
            columns[name] = header
            picked = [i for i, h in enumerate(header) if any(w in h.lower() for w in LABEL_WORDS)]
            for index in picked:
                counts = Counter(row[index].strip() for row in body if len(row) > index)
                for value, count in sorted(counts.items()):
                    rows.append([relative(folder, root), path.name, header[index], value, count])
            if not picked:
                rows.append([relative(folder, root), path.name, "", "", len(body)])
            best = None
            sample = body[:REFERENCE_ROWS]
            for index, title in enumerate(header[:50]):
                values = [row[index].strip() for row in sample if len(row) > index]
                values = [v for v in values if v]
                hits = sum(PurePosixPath(v).name in image_names or v in image_stems for v in values)
                filenames = [v for v in values if PurePosixPath(v).suffix.lower() in IMAGE_SUFFIXES]
                missing = sum(PurePosixPath(v).name not in image_names for v in filenames)
                if hits and (best is None or hits > best["matched"]):
                    best = {
                        "column": title,
                        "rows": len(values),
                        "matched": hits,
                        "missing": missing,
                    }
            references[name] = best
    return rows, columns, references


def field_report(folders, root: Path) -> list[dict]:
    by_depth = defaultdict(set)
    for folder in folders:
        parts = folder.relative_to(root).parts
        if parts:
            by_depth[len(parts)].add(parts[-1])
    report = []
    for depth, names in sorted(by_depth.items()):
        split = [SEPARATORS.split(name) for name in sorted(names)]
        positions = []
        for index in range(max(len(s) for s in split)):
            values = sorted({s[index] for s in split if len(s) > index})
            positions.append(
                {"position": index + 1, "distinct": len(values), "examples": values[:12]}
            )
        lengths = Counter(len(s) for s in split)
        report.append(
            {
                "depth": depth,
                "folders": len(names),
                "field_counts": dict(lengths),
                "fields": positions,
            }
        )
    return report


def name_patterns(folders) -> Counter:
    patterns = Counter()
    for files in folders.values():
        for path in files:
            patterns[re.sub(r"\d+", "#", path.name)] += 1
    return patterns


def inspect(root: Path, config: dict) -> dict:
    folders = walk(root)
    tree = tree_rows(folders, root)
    images = {
        f: [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES] for f, files in folders.items()
    }
    images = {f: files for f, files in images.items() if files}
    roles = {f: mask_reason(f, files, config) for f, files in images.items()}
    headers, image_rows, issues = {}, [], []
    for folder, files in images.items():
        found = {p: header_of(p) for p in files}
        headers.update(found)
        unreadable = sum(v is None for v in found.values())
        kinds = Counter(f"{w}x{h}x{m}" for w, h, m in (v for v in found.values() if v))
        listed = ";".join(f"{k}:{n}" for k, n in sorted(kinds.items()))
        role = "masks" if roles[folder] else "images"
        image_rows.append(
            [relative(folder, root), role, len(files), len(kinds), listed, unreadable]
        )
        if unreadable:
            issues.append(f"{relative(folder, root)}: {unreadable} images could not be opened")
        if role == "images" and len(kinds) > 1:
            issues.append(f"{relative(folder, root)}: {len(kinds)} different size and mode values")
    for folder, files in folders.items():
        empty = [f for f in files if f.stat().st_size == 0]
        if empty:
            issues.append(f"{relative(folder, root)}: {len(empty)} empty files")
    photo_index = defaultdict(list)
    for folder, files in images.items():
        if not roles[folder]:
            for path in files:
                photo_index[normalized(path.stem)].append(path)
    mask_rows, mask_facts, matched_images = [], [], set()
    for folder, files in images.items():
        if not roles[folder]:
            continue
        facts = summarize_masks(files)
        paired, sources, mismatched, ambiguous = 0, Counter(), 0, 0
        for path in files:
            candidates = photo_index.get(normalized(path.stem), [])
            if not candidates:
                continue
            if len(candidates) > 1:
                ambiguous += 1
            image = max(candidates, key=lambda c: (common_depth(c.parent, folder), str(c)))
            paired += 1
            matched_images.add(image)
            sources[relative(image.parent, root)] += 1
            mask_size, image_size = headers.get(path), headers.get(image)
            if mask_size and image_size and mask_size[:2] != image_size[:2]:
                mismatched += 1
        facts.update(
            {
                "folder": relative(folder, root),
                "detected_by": roles[folder],
                "paired": paired,
                "unpaired": len(files) - paired,
                "ambiguous": ambiguous,
                "size_mismatch": mismatched,
                "image_folders": sources,
            }
        )
        mask_facts.append(facts)
        top = sorted(facts["values"].items(), key=lambda item: -item[1])[:20]
        mask_rows.append(
            [
                facts["folder"],
                facts["detected_by"],
                facts["masks"],
                len(facts["values"]),
                ";".join(f"{v}:{n}" for v, n in top),
                facts["all_zero_share"],
                facts["multi_value_share"],
                facts["components_min"],
                facts["components_median"],
                facts["components_mean"],
                facts["components_max"],
                facts["multi_component_share"],
                paired,
                facts["size_mismatch"],
            ]
        )
        if len(files) - paired:
            issues.append(
                f"{facts['folder']}: {len(files) - paired} masks without a matching image"
            )
        if mismatched:
            issues.append(f"{facts['folder']}: {mismatched} masks differ in size from their image")
        if facts["unreadable"]:
            issues.append(f"{facts['folder']}: {facts['unreadable']} masks could not be opened")
    annotated = {image.parent for image in matched_images}
    for folder in sorted(annotated):
        missing = [p for p in images[folder] if p not in matched_images]
        if missing:
            issues.append(f"{relative(folder, root)}: {len(missing)} images without a mask")
    photos = [p for f, files in images.items() if not roles[f] for p in files]
    names = {p.name for p in photos}
    stems = {p.stem for p in photos}
    label_rows, csv_columns, references = label_tables(folders, root, names, stems)
    for name, found in references.items():
        if found and found["missing"]:
            issues.append(
                f"{name}: {found['missing']} file names in {found['column']} do not exist"
            )
    others = Counter(
        p.suffix.lower()
        for files in folders.values()
        for p in files
        if p.suffix.lower() in TABLE_SUFFIXES
    )
    if others:
        listed = ", ".join(f"{n} {s}" for s, n in sorted(others.items()))
        issues.append(f"tables not read by this job: {listed}")
    return {
        "tree": tree,
        "images": image_rows,
        "labels": label_rows,
        "masks": mask_rows,
        "mask_facts": mask_facts,
        "csv_columns": csv_columns,
        "references": references,
        "fields": field_report(folders, root),
        "patterns": name_patterns(folders),
        "issues": issues,
        "n_images": len(photos),
        "n_masks": sum(len(images[f]) for f in images if roles[f]),
    }


def mask_reading(facts: list[dict]) -> list[str]:
    if not facts:
        return ["No mask folder was found: no image folder holds images with few distinct values."]
    values = set()
    for item in facts:
        values |= {v for v in item["values"] if not zero_label(v)}
    lines = []
    if len(values) <= 1:
        only = next(iter(values), "none")
        lines.append(
            f"Masks encode foreground only: every mask uses the single non-zero value {only}, "
            "so the defect class is not stored per pixel."
        )
    else:
        multi = [item["multi_value_share"] or 0 for item in facts]
        lines.append(
            f"Masks use {len(values)} distinct non-zero values ({', '.join(sorted(values)[:12])}). "
            f"The share of non-empty masks holding more than one non-zero value is "
            f"{max(multi):.4f} at most per folder. A value per class would show as one value "
            "per defect type; compare the values with the labels before relying on them."
        )
    for item in facts:
        multi = item["multi_component_share"]
        equal = item["values_equal_components_share"]
        text = (
            f"{item['folder']}: {item['masks']} masks, {item['all_zero_share']} all zero, "
            f"{item['components_min']} to {item['components_max']} connected components per "
            f"non-empty mask (median {item['components_median']}), {multi} with more than one."
        )
        if multi and equal:
            text += f" In {equal} of those the number of values equals the number of components."
        lines.append(text)
    lines.append(
        "Separate defects are separable as instances only as connected components unless the "
        "values above differ per defect; touching defects would merge."
    )
    return lines


def summary_text(record: dict, found: dict, downloads: list, skipped: list) -> str:
    license_ = record.get("license") or {}
    lines = [
        "# TSFabrics fetch",
        "",
        f"Record: {record.get('title')}, version {record.get('version')}, DOI {record.get('doi')}.",
        f"License as stated by Figshare: {license_.get('name')} ({license_.get('url')}).",
        "",
        "## Download",
        "",
    ]
    for name, size, status in downloads:
        lines.append(f"- {name}: {size} bytes, md5 matches the published value ({status})")
    lines += [
        "",
        "## Contents",
        "",
        f"- Folders: {len(found['tree'])}, image files: {found['n_images']}, "
        f"mask files: {found['n_masks']}",
        f"- Image folders: {sum(r[1] == 'images' for r in found['images'])}, "
        f"mask folders: {sum(r[1] == 'masks' for r in found['images'])}",
        "",
        "## Masks",
        "",
        *[f"- {line}" for line in mask_reading(found["mask_facts"])],
        "",
        "## Pairing",
        "",
    ]
    for item in found["mask_facts"]:
        where = ", ".join(f"{f} ({n})" for f, n in item["image_folders"].most_common(3)) or "none"
        lines.append(
            f"- {item['folder']}: {item['paired']} of {item['masks']} masks share a file stem with "
            f"an image (ignoring words like mask or label), {item['ambiguous']} ambiguous; "
            f"images from {where}"
        )
    if not found["mask_facts"]:
        lines.append("- No masks, so no image and mask pairing.")
    for name, found_ref in found["references"].items():
        if found_ref:
            lines.append(
                f"- {name}: column {found_ref['column']} names an existing image in "
                f"{found_ref['matched']} of {found_ref['rows']} rows"
            )
        else:
            lines.append(f"- {name}: no column names an image file")
    lines += ["", "## Labels", ""]
    if found["csv_columns"]:
        for name, header in found["csv_columns"].items():
            lines.append(f"- {name}: columns {', '.join(header[:20]) or 'none'}")
        label_columns = {(r[1], r[2]) for r in found["labels"] if r[2]}
        if not label_columns:
            lines.append("- No CSV column name mentions label, class, defect or type.")
    else:
        lines.append("- No CSV file found, so no image level labels.")
    lines += ["", "## Folder name fields", ""]
    for level in found["fields"]:
        fields = "; ".join(
            f"field {f['position']}: {f['distinct']} values (e.g. {', '.join(f['examples'][:6])})"
            for f in level["fields"]
        )
        lines.append(
            f"- Depth {level['depth']}: {level['folders']} folder names, fields per name "
            f"{level['field_counts']}; {fields}"
        )
    top = ", ".join(f"{p} ({n})" for p, n in found["patterns"].most_common(8))
    lines += [f"- File name patterns, digits as #: {top}", "", "## Grouped split", ""]
    groups = [level for level in found["fields"] if level["folders"] > 1]
    if groups:
        for level in groups:
            lines.append(
                f"- Depth {level['depth']} has {level['folders']} distinct folders, a candidate "
                "grouping unit if each folder is one capture run, fabric or machine setting."
            )
    else:
        lines.append("- No folder level has more than one folder, so folders give no grouping.")
    lines += ["", "## Inconsistencies", ""]
    lines += [f"- {issue}" for issue in found["issues"]] or ["- None found."]
    for path in skipped:
        lines.append(f"- Nested archive not extracted (depth limit): {path.name}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_results(out_dir: Path, record: dict, entries: list[dict], found: dict, text: str):
    kept = {k: record.get(k) for k in ("id", "title", "doi", "version", "license")}
    kept |= {k: record.get(k) for k in ("published_date", "url_public_html")}
    kept["files"] = [
        {k: e[k] for k in ("name", "size", "md5", "supplied_md5", "url")} | {"md5_verified": True}
        for e in entries
    ]
    (out_dir / "record.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")
    write_csv(out_dir / "tree.csv", ["folder", "files", "extensions", "bytes"], found["tree"])
    write_csv(
        out_dir / "images.csv",
        ["folder", "role", "count", "distinct", "width_x_height_x_mode", "unreadable"],
        found["images"],
    )
    write_csv(
        out_dir / "labels.csv", ["folder", "csv", "column", "value", "count"], found["labels"]
    )
    write_csv(
        out_dir / "masks.csv",
        [
            "folder",
            "detected_by",
            "masks",
            "distinct_values",
            "value_pixel_counts",
            "all_zero_share",
            "multi_value_share",
            "components_min",
            "components_median",
            "components_mean",
            "components_max",
            "multi_component_share",
            "paired_with_image",
            "size_mismatch",
        ],
        found["masks"],
    )
    (out_dir / "summary.md").write_text(text, encoding="utf-8")


def file_entries(record: dict) -> list[dict]:
    entries = []
    for item in record.get("files", []):
        if item.get("is_link_only"):
            raise SystemExit(f"{item['name']} is a link only file and cannot be verified")
        entries.append(
            {
                "name": item["name"],
                "size": int(item["size"]),
                "md5": item.get("computed_md5") or item.get("supplied_md5"),
                "supplied_md5": item.get("supplied_md5"),
                "url": item["download_url"],
            }
        )
    if not entries:
        raise SystemExit("the Figshare record lists no files")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = runio.load_config(args.config)
    if config.get("runs_root"):
        os.environ["FABRID_RUNS"] = str(config["runs_root"])
    out_dir = runio.start_run(config["step"], config["name"], config)
    heavy = runio.runs_dir(config["step"], config["name"])
    record = resolve(config["api"], config["doi"], config["timeout"])
    entries = file_entries(record)
    print(f"record {record.get('id')}: {record.get('title')}, version {record.get('version')}")
    for entry in entries:
        print(f"  {entry['name']}: {entry['size']} bytes, md5 {entry['md5']}")
    margin = config["margin_gb"] * GB
    remaining = 0
    for entry in entries:
        partial = heavy / "download" / f"{entry['name']}.part"
        done = heavy / "download" / entry["name"]
        if not done.is_file():
            remaining += entry["size"] - (partial.stat().st_size if partial.is_file() else 0)
    archived = sum(e["size"] for e in entries if is_archive(Path(e["name"])))
    check_space(heavy, remaining + archived * config["expansion"] + margin)
    downloads = []
    for entry in entries:
        status = download(
            entry, heavy / "download", config["retries"], config["retry_wait"], config["timeout"]
        )
        print(f"{entry['name']}: {status}, md5 verified")
        downloads.append((entry["name"], entry["size"], status))
    archives = [heavy / "download" / e["name"] for e in entries if is_archive(Path(e["name"]))]
    extracted = heavy / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    _, skipped = extract_all(archives, extracted, config["nested_depth"], margin)
    for entry in entries:
        if not is_archive(Path(entry["name"])):
            source, target = heavy / "download" / entry["name"], extracted / entry["name"]
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
    found = inspect(extracted, config)
    text = summary_text(record, found, downloads, skipped)
    write_results(out_dir, record, entries, found, text)
    print(text)
    print(f"wrote {out_dir}, data in {heavy}")
    if config.get("publish", True):
        runio.publish(out_dir)


if __name__ == "__main__":
    main()
