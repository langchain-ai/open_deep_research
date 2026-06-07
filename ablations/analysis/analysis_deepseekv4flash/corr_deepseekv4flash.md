# DeepSeek v4 Flash — Citation Retention vs Regression (v2 → v3)

> **citation_retention_pct**: fraction of v2 URLs still present in v3 report  
> **regression_rate**: mean per-category regression rate averaged across 4 rubric axes  
> **regression_count**: total criteria that regressed (sat→unsat) across all 4 axes  
> Tasks marked † had no v3 eval — v2 used as fallback (regression=0, retention=100%)

| task_id        | citation_retention_% | regression_rate_% | regression_count |
| -------------- | --------------------: | -----------------: | ----------------: |
| task_001       |                97.78 |              0.93 |                1 |
| task_002       |                67.65 |             15.71 |                4 |
| task_003       |                87.39 |              2.08 |                2 |
| task_004       |                90.48 |              6.94 |                3 |
| task_006       |                39.13 |             10.88 |                5 |
| task_008       |                34.15 |             18.69 |                5 |
| task_011†      |               100.00 |              0.00 |                0 |
| task_012       |                68.00 |              6.25 |                1 |
| task_014       |                  N/A |              9.17 |                3 |
| task_015       |                49.12 |              8.52 |                2 |
| task_016       |                43.40 |             12.50 |                2 |
| task_018       |                11.11 |             19.77 |                5 |
| task_019†      |               100.00 |              0.00 |                0 |
| task_021       |                41.84 |              2.27 |                1 |
| task_023       |                70.49 |              4.17 |                1 |
| task_028       |                16.42 |             18.57 |                6 |
| task_031       |                14.49 |              8.33 |                1 |
| task_032       |                51.69 |              9.38 |                2 |
| task_034       |                47.25 |              7.14 |                2 |
| task_035       |                16.18 |             13.39 |                4 |
| task_036       |                16.36 |             44.88 |               12 |
| task_039       |                62.07 |             24.94 |                6 |
| task_044       |                66.67 |              2.78 |                2 |
| task_045       |                40.00 |              7.27 |                2 |
| task_050       |                48.15 |              1.67 |                1 |
| task_052       |                43.06 |             16.67 |                5 |
| task_053       |                60.26 |              6.25 |                1 |
| task_055       |                38.30 |             21.67 |                3 |
| task_056       |                60.16 |             12.50 |                3 |
| task_058       |                25.97 |              3.57 |                1 |
| task_061       |                47.62 |             22.92 |                3 |
| task_063       |                71.74 |              1.32 |                1 |
| task_066       |                13.89 |              4.17 |                2 |
| task_068       |                25.00 |              7.42 |                3 |
| task_070†      |               100.00 |              0.00 |                0 |
| task_071       |                72.73 |             13.75 |                4 |
| task_073       |                85.00 |              5.00 |                1 |
| task_078       |                23.29 |              6.25 |                1 |
| task_079       |                50.94 |             15.71 |                4 |
| task_080       |                25.93 |              1.39 |                1 |
| task_084       |                60.81 |             10.00 |                4 |
| task_086       |                30.19 |             18.75 |                5 |
| task_087       |                52.34 |              2.08 |                1 |
| task_088       |                56.00 |              0.00 |                0 |
| task_089       |                70.83 |              0.00 |                0 |
| task_090       |                98.18 |              0.00 |                0 |
| task_092       |                63.33 |              0.00 |                0 |
| task_095       |                62.32 |              1.79 |                2 |
| task_096       |                72.34 |              6.25 |                3 |
| task_098       |                  N/A |              7.50 |                2 |

## Averages

| Metric | Mean |
| :----- | ---: |
| citation_retention_% | 53.96 |
| regression_rate_%    | 8.82 |
| regression_count     | 2.46 |