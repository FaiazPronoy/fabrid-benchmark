# Pilot on new fabrics

Pilot test: 93 images from 5 fabrics (black_white_check, blue_check, maroon_floral, olive, royal_blue).
Single endpoint: mask AP50 over the four classes. Intervals are 95% bootstraps over the pilot test images, which come from only 5 fabrics; they do not describe variation between fabrics.

| arm | mask AP50 | box AP50 |
|---|---|---|
| pilot_old | 0.133 [0.081, 0.236] | 0.134 [0.081, 0.235] |
| pilot_new | 0.293 [0.223, 0.396] | 0.300 [0.233, 0.399] |
| pilot_merged | 0.326 [0.252, 0.445] | 0.351 [0.276, 0.467] |

| comparison | mask AP50 difference | verdict |
|---|---|---|
| pilot_new minus pilot_old | 0.160 [0.064, 0.236] | better on new fabrics |
| pilot_merged minus pilot_old | 0.193 [0.113, 0.279] | better on new fabrics |
| pilot_merged minus pilot_new | 0.032 [-0.028, 0.117] | not shown |

Rule: a candidate counts as better than its baseline on new fabrics only if the lower end of the paired 95% interval of its mask AP50 difference is above 0. Anything else is reported as not shown.
