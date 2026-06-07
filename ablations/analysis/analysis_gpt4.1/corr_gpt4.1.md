# GPT-4.1 — Citation Retention vs Regression (v2 → v3)

> **citation_retention_pct**: fraction of v2 URLs still present in v3 report  
> **regression_rate**: mean per-category regression rate averaged across 4 rubric axes  
> **regression_count**: total criteria that regressed (sat→unsat) across all 4 axes  
> Tasks marked † had no v3 eval — v2 used as fallback (regression=0, retention=100%)

| task_id        | citation_retention_% | regression_rate_% | regression_count |
| -------------- | --------------------: | -----------------: | ----------------: |
| task_001       |                51.85 |             20.83 |               16 |
| task_002       |                25.00 |             16.67 |                2 |
| task_003       |                29.41 |             29.55 |               10 |
| task_004       |                50.00 |             68.75 |                5 |
| task_006       |                29.17 |             22.60 |                6 |
| task_008       |                28.12 |             27.08 |                7 |
| task_011       |                15.79 |             53.12 |               11 |
| task_012       |                22.73 |             15.62 |                5 |
| task_014       |                13.04 |             38.33 |                9 |
| task_015       |                14.29 |             19.79 |                7 |
| task_016       |                16.67 |             29.17 |               11 |
| task_018       |                17.14 |             10.71 |                3 |
| task_019       |                 2.50 |             25.85 |               14 |
| task_021       |                 7.14 |             28.06 |                9 |
| task_023       |                15.00 |             11.11 |                4 |
| task_028       |                12.12 |             39.74 |               14 |
| task_031       |                30.00 |             22.73 |                4 |
| task_032       |                12.50 |             47.62 |               10 |
| task_034       |                37.29 |             35.71 |                8 |
| task_035       |                26.32 |             18.33 |                3 |
| task_036       |                 0.00 |             29.17 |                4 |
| task_039       |                47.06 |             21.71 |                9 |
| task_044       |                20.00 |             17.65 |                6 |
| task_045       |                22.35 |             26.93 |                6 |
| task_050       |                 8.47 |             31.67 |               10 |
| task_052       |                28.12 |             18.12 |                6 |
| task_053       |                66.00 |             10.42 |                2 |
| task_055       |                21.05 |             22.92 |                3 |
| task_056       |                40.82 |             18.57 |                4 |
| task_058       |                24.44 |              6.25 |                2 |
| task_061       |                50.00 |             11.11 |                1 |
| task_063       |                33.33 |              3.75 |                3 |
| task_066       |                11.43 |              9.38 |                2 |
| task_068       |                 7.14 |             19.05 |                6 |
| task_070       |                66.67 |              0.00 |                0 |
| task_071       |                20.00 |              8.33 |                1 |
| task_073       |                51.52 |              8.57 |                4 |
| task_078       |                 0.00 |             17.27 |                4 |
| task_079       |                18.18 |             31.67 |                7 |
| task_080       |                38.89 |             20.25 |                8 |
| task_084       |                22.95 |              9.17 |                2 |
| task_086       |                25.00 |             20.83 |                3 |
| task_087       |                60.34 |              8.33 |                2 |
| task_088       |                26.56 |             13.33 |                2 |
| task_089       |                11.11 |             39.29 |                7 |
| task_090       |                60.00 |              5.00 |                1 |
| task_092       |                31.58 |             15.00 |                3 |
| task_095       |                34.00 |              6.01 |                4 |
| task_096       |                37.50 |             27.08 |                5 |
| task_098       |                10.00 |             47.14 |                6 |

## Averages

| Metric | Mean |
| :----- | ---: |
| citation_retention_% | 27.01 |
| regression_rate_%    | 22.11 |
| regression_count     | 5.62 |