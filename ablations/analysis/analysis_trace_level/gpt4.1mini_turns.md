# gpt4.1mini — Turns Analysis (v1 / v2 / v3)


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| mini_turn1             |   50 |        908,548 |         56,412 |   $19.9884 |       243.7s |
| mini_turn2             |   50 |      1,183,568 |         72,097 |   $25.6203 |       209.4s |
| mini_turn3             |   50 |      1,335,201 |         81,017 |   $27.5895 |       327.8s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| mini_turn1             |    2.5 / 3.0 / 6 | 11.8 / 12.0 / 30 | 5.5 / 6.0 / 21 | 110.5 / 96.5 / 294 |
| mini_turn2             |    2.7 / 3.0 / 6 | 13.2 / 10.5 / 44 | 7.0 / 5.0 / 38 | 139.1 / 121.0 / 356 |
| mini_turn3             |    3.0 / 3.0 / 8 | 14.7 / 15.0 / 35 | 8.1 / 7.0 / 30 | 149.8 / 141.5 / 465 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| mini_turn1             |          $0.3030 (75.8%) |         $0.0867 (21.68%) |          $0.0006 (0.15%) |          $0.0095 (2.37%) |
| mini_turn2             |         $0.3871 (75.54%) |         $0.1099 (21.46%) |          $0.0023 (0.45%) |          $0.0131 (2.55%) |
| mini_turn3             |         $0.4215 (73.34%) |         $0.1366 (23.77%) |          $0.0027 (0.46%) |          $0.0140 (2.43%) |

## Report Characteristics
> mean per task across all tasks with that version

| Metric             |    v1 mean |    v2 mean |    Δ v1→v2 |    v3 mean |    Δ v2→v3 |
| :----------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| word_count         |     2052.2 |     2549.2 |     +497.0 |     2527.1 |      -22.1 |
| citation_count     |       19.6 |       22.0 |       +2.4 |       21.5 |       -0.4 |
| section_count      |       21.7 |       24.8 |       +3.0 |       24.8 |       +0.1 |