import csv
import hashlib
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from fabrid import fetch

DOI = "10.0000/test.figshare.7"


class Server:
    def __init__(self, files: dict[str, bytes], md5: dict[str, str] | None = None):
        self.files = files
        self.md5 = md5 or {}
        self.cut = {}
        self.ranges = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def send_json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                query = json.loads(self.rfile.read(length))
                hits = [{"id": 7, "doi": f"{DOI}.v1"}] if query.get("doi") == DOI else []
                self.send_json(hits)

            def do_GET(self):
                if self.path == "/api/articles/7":
                    self.send_json(outer.record())
                    return
                name = self.path.removeprefix("/files/")
                data = outer.files[name]
                wanted = self.headers.get("Range")
                outer.ranges.append(wanted)
                start = int(wanted.split("=")[1].rstrip("-")) if wanted else 0
                self.send_response(206 if wanted else 200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                stop = outer.cut.pop(name, len(data))
                self.wfile.write(data[start:stop])
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def record(self):
        files = [
            {
                "id": index,
                "name": name,
                "size": len(data),
                "computed_md5": self.md5.get(name, hashlib.md5(data).hexdigest()),
                "supplied_md5": hashlib.md5(data).hexdigest(),
                "download_url": f"{self.base}/files/{name}",
                "is_link_only": False,
            }
            for index, (name, data) in enumerate(self.files.items())
        ]
        license_ = {
            "value": 1,
            "name": "CC BY",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        }
        return {
            "id": 7,
            "title": "Test set",
            "doi": f"{DOI}.v1",
            "version": 1,
            "license": license_,
            "files": files,
        }


@pytest.fixture
def served():
    started = []

    def make(files, md5=None):
        server = Server(files, md5)
        started.append(server)
        return server

    yield make
    for server in started:
        server.httpd.shutdown()


def entry_for(server: Server, name: str) -> dict:
    return fetch.file_entries(server.record())[
        [f["name"] for f in server.record()["files"]].index(name)
    ]


def save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def synthetic_tree(root: Path) -> None:
    rng = np.random.default_rng(0)
    for fabric in ("rib_hole_a", "rib_stain_b"):
        for frame in range(3):
            photo = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
            save(root / "data" / fabric / "images" / f"frame_{frame:03d}.png", photo)
    two_blobs = np.zeros((24, 32), dtype=np.uint8)
    two_blobs[2:5, 2:5] = 255
    two_blobs[15:20, 20:25] = 255
    one_blob = np.zeros((24, 32), dtype=np.uint8)
    one_blob[8:12, 8:14] = 255
    empty = np.zeros((24, 32), dtype=np.uint8)
    hole = root / "data" / "rib_hole_a" / "masks"
    save(hole / "frame_000_mask.png", two_blobs)
    save(hole / "frame_001_mask.png", one_blob)
    save(hole / "frame_002_mask.png", empty)
    save(root / "data" / "rib_stain_b" / "masks" / "frame_000_mask.png", one_blob)
    save(root / "data" / "rib_stain_b" / "masks" / "frame_009_mask.png", one_blob)
    with (root / "data" / "rib_hole_a" / "labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "defect"])
        writer.writerows(
            [
                ["frame_000.png", "hole"],
                ["frame_001.png", "hole"],
                ["frame_002.png", "ok"],
                ["frame_007.png", "hole"],
            ]
        )


def zipped(root: Path) -> bytes:
    archive = root.parent / "tree.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    return archive.read_bytes()


CONFIG = {"mask_sample": 20, "mask_max_values": 32}


def test_resolve_by_doi(served):
    server = served({"a.bin": b"x"})
    record = fetch.resolve(f"{server.base}/api", DOI)
    assert record["title"] == "Test set"
    with pytest.raises(SystemExit, match="no record"):
        fetch.resolve(f"{server.base}/api", "10.0000/other")


def test_checksum_mismatch_is_refused(served, tmp_path):
    server = served({"a.bin": b"payload" * 100}, {"a.bin": "0" * 32})
    with pytest.raises(SystemExit, match="does not match"):
        fetch.download(entry_for(server, "a.bin"), tmp_path, 2, 0)
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "a.bin.part").exists()


def test_download_resumes_after_interruption(served, tmp_path):
    data = bytes(range(256)) * 400
    server = served({"a.bin": data})
    server.cut["a.bin"] = 30_000
    status = fetch.download(entry_for(server, "a.bin"), tmp_path, 3, 0)
    assert status == "resumed"
    assert (tmp_path / "a.bin").read_bytes() == data
    assert server.ranges == [None, "bytes=30000-"]
    assert fetch.download(entry_for(server, "a.bin"), tmp_path, 3, 0) == "present"
    assert len(server.ranges) == 2


def test_partial_file_from_earlier_run_is_resumed(served, tmp_path):
    data = b"abcdefgh" * 1000
    server = served({"a.bin": data})
    (tmp_path / "a.bin.part").write_bytes(data[:5000])
    assert fetch.download(entry_for(server, "a.bin"), tmp_path, 2, 0) == "resumed"
    assert server.ranges == ["bytes=5000-"]
    assert (tmp_path / "a.bin").read_bytes() == data


def test_interrupted_extraction_is_redone(tmp_path):
    tree = tmp_path / "tree"
    synthetic_tree(tree)
    archive = tmp_path / "set.zip"
    archive.write_bytes(zipped(tree))
    dest = tmp_path / "out" / "set"
    dest.mkdir(parents=True)
    (dest / "half_written.png").write_bytes(b"junk")
    assert fetch.extract(archive, dest) == "redone"
    assert not (dest / "half_written.png").exists()
    assert (dest / fetch.MARKER).is_file()
    assert (dest / "data" / "rib_hole_a" / "labels.csv").is_file()
    stamp = (dest / fetch.MARKER).stat().st_mtime_ns
    assert fetch.extract(archive, dest) == "present"
    assert (dest / fetch.MARKER).stat().st_mtime_ns == stamp


def test_extraction_refuses_escaping_paths_and_low_disk(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "x")
    with pytest.raises(SystemExit, match="escapes"):
        fetch.extract(archive, tmp_path / "out" / "bad")
    with pytest.raises(SystemExit, match="not enough free disk"):
        fetch.check_space(tmp_path, 10 * fetch.GB, free=fetch.GB)
    fetch.check_space(tmp_path, 1, free=fetch.GB)


def test_inspection_tables(tmp_path):
    root = tmp_path / "root"
    synthetic_tree(root)
    found = fetch.inspect(root, CONFIG)
    tree = {row[0]: row for row in found["tree"]}
    assert tree["data/rib_hole_a"][1:3] == [1, "csv:1"]
    assert tree["data/rib_hole_a/masks"][1:3] == [3, "png:3"]
    images = {row[0]: row for row in found["images"]}
    assert images["data/rib_hole_a/images"][1:5] == ["images", 3, 1, "32x24xRGB:3"]
    assert images["data/rib_hole_a/masks"][1] == "masks"
    masks = {row[0]: row for row in found["masks"]}
    hole = masks["data/rib_hole_a/masks"]
    assert hole[2] == 3
    assert hole[3] == 2
    assert hole[4] == f"0:{24 * 32 * 3 - 9 - 25 - 24};255:{9 + 25 + 24}"
    assert hole[5] == pytest.approx(1 / 3, abs=1e-4)
    assert hole[7:11] == [1, 1.5, 1.5, 2]
    assert hole[11] == 0.5
    assert hole[12:14] == [3, 0]
    stain = masks["data/rib_stain_b/masks"]
    assert stain[12] == 1
    labels = {(row[2], row[3]): row[4] for row in found["labels"]}
    assert labels == {("defect", "hole"): 3, ("defect", "ok"): 1}
    reference = found["references"]["data/rib_hole_a/labels.csv"]
    assert reference == {"column": "file", "rows": 4, "matched": 3, "missing": 1}
    issues = " ".join(found["issues"])
    assert "data/rib_stain_b/masks: 1 masks without a matching image" in issues
    assert "data/rib_stain_b/images: 2 images without a mask" in issues
    assert "1 file names in file do not exist" in issues
    reading = " ".join(fetch.mask_reading(found["mask_facts"]))
    assert "foreground only" in reading
    depth2 = next(level for level in found["fields"] if level["depth"] == 2)
    assert depth2["folders"] == 2
    assert depth2["fields"][1] == {"position": 2, "distinct": 2, "examples": ["hole", "stain"]}
    assert found["patterns"]["frame_#.png"] == 6


def test_main_end_to_end(served, tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    synthetic_tree(tree)
    server = served({"set.zip": zipped(tree)})
    config = {
        "step": "external",
        "name": "tsfabrics_fetch",
        "doi": DOI,
        "api": f"{server.base}/api",
        "runs_root": str(tmp_path / "runs"),
        "timeout": 10,
        "retries": 2,
        "retry_wait": 0,
        "expansion": 1.1,
        "margin_gb": 0,
        "nested_depth": 2,
        **CONFIG,
        "publish": False,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("FABRID_REPO", str(tmp_path / "repo"))
    monkeypatch.setattr("sys.argv", ["fetch", "--config", str(path)])
    fetch.main()
    out = tmp_path / "repo" / "results" / "external" / "tsfabrics_fetch"
    names = ("config.yaml", "record.json", "tree.csv", "images.csv", "labels.csv", "masks.csv")
    assert all((out / name).is_file() for name in (*names, "summary.md"))
    record = json.loads((out / "record.json").read_text(encoding="utf-8"))
    assert record["license"]["name"] == "CC BY"
    assert record["files"][0]["md5_verified"] is True
    assert "md5 matches" in (out / "summary.md").read_text(encoding="utf-8")
    assert (
        tmp_path / "runs" / "external" / "tsfabrics_fetch" / "extracted" / "set" / fetch.MARKER
    ).is_file()
