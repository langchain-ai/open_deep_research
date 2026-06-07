# GPT-4.1-mini — Citation Retention vs Regression (v2 → v3)

> **citation_retention_pct**: fraction of v2 URLs still present in v3 report  
> **regression_rate**: mean per-category regression rate averaged across 4 rubric axes  
> **regression_count**: total criteria that regressed (sat→unsat) across all 4 axes  
> Tasks marked † had no v3 eval — v2 used as fallback (regression=0, retention=100%)

| task_id        | citation_retention_% | regression_rate_% | regression_count |
| -------------- | --------------------: | -----------------: | ----------------: |
| task_001       |                31.25 |             14.56 |                8 |
| task_002       |                50.00 |             36.25 |                3 |
| task_003       |                41.18 |             20.42 |                4 |
| task_004       |                31.25 |              6.25 |                1 |
| task_006       |                30.00 |              9.55 |                3 |
| task_008       |                52.94 |              3.57 |                1 |
| task_011       |                30.00 |             34.44 |                9 |
| task_012       |                27.78 |             18.89 |                6 |
| task_014       |                33.33 |             17.50 |                5 |
| task_015       |                18.75 |             13.21 |                4 |
| task_016       |                42.86 |             15.48 |                3 |
| task_018       |                38.46 |             24.40 |                5 |
| task_019       |                 3.85 |             13.45 |                8 |
| task_021       |                31.03 |             55.00 |               14 |
| task_023       |                80.00 |             12.50 |                4 |
| task_028       |                22.22 |             11.31 |                4 |
| task_031       |                45.45 |              4.55 |                2 |
| task_032       |                25.00 |             15.62 |                6 |
| task_034       |                40.74 |             17.50 |                5 |
| task_035       |                27.27 |              6.25 |                1 |
| task_036       |                30.00 |             22.14 |                8 |
| task_039       |                50.00 |              0.00 |                0 |
| task_044       |                23.08 |              9.55 |                3 |
| task_045       |                30.00 |             22.50 |                6 |
| task_050       |                13.33 |             10.42 |                3 |
| task_052       |                28.95 |             27.50 |                6 |
| task_053       |                62.50 |              8.33 |                1 |
| task_055       |                50.00 |             17.86 |                5 |
| task_056       |                29.17 |             35.24 |                6 |
| task_058       |                41.94 |              3.57 |                1 |
| task_061       |                58.33 |             38.89 |                3 |
| task_063       |                33.33 |             16.99 |                4 |
| task_066       |                17.65 |             10.62 |                4 |
| task_068       |                30.77 |             18.33 |                3 |
| task_070†      |               100.00 |              0.00 |                0 |
| task_071       |                35.71 |             19.58 |                3 |
| task_073       |                69.23 |             20.06 |                8 |
| task_078       |                15.00 |             17.50 |                4 |
| task_079       |                41.67 |             23.21 |                5 |
| task_080       |                46.15 |              8.78 |                6 |
| task_084       |                27.50 |             25.00 |                4 |
| task_086       |                37.50 |             13.33 |                2 |
| task_087       |                18.18 |             26.25 |                5 |
| task_088       |                29.03 |              0.00 |                0 |
| task_089       |                36.36 |             15.87 |                2 |
| task_090†      |               100.00 |              0.00 |                0 |
| task_092       |                36.36 |             10.00 |                2 |
| task_095       |                10.42 |             18.38 |               12 |
| task_096       |                29.03 |             18.12 |                4 |
| task_098       |                26.67 |             41.67 |                5 |

## Averages

| Metric | Mean |
| :----- | ---: |
| citation_retention_% | 37.22 |
| regression_rate_%    | 17.01 |
| regression_count     | 4.22 |