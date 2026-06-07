# deepseekv4flash — Turns Analysis (v1 / v2 / v3)


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| deepseek_turn1         |   50 |      2,557,090 |        156,617 |   $35.2076 |       459.6s |
| deepseek_turn2         |   50 |      3,854,629 |        217,629 |   $46.2459 |       789.0s |
| deepseek_turn3         |   47 |      4,040,899 |        248,395 |   $50.9798 |       683.1s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| deepseek_turn1         |    4.2 / 4.0 / 8 | 36.2 / 35.0 / 80 | 23.4 / 21.5 / 73 | 262.9 / 279.5 / 466 |
| deepseek_turn2         |   5.7 / 5.5 / 10 | 46.4 / 45.0 / 85 | 31.8 / 28.0 / 91 | 630.7 / 414.5 / 2282 |
| deepseek_turn3         |   6.0 / 6.0 / 12 | 51.9 / 50.0 / 96 | 35.5 / 33.0 / 85 | 369.3 / 331.0 / 767 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| deepseek_turn1         |         $0.7042 (100.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |
| deepseek_turn2         |         $0.9249 (100.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |
| deepseek_turn3         |         $1.0847 (100.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |

## Report Characteristics
> mean per task across all tasks with that version

| Metric             |    v1 mean |    v2 mean |    Δ v1→v2 |    v3 mean |    Δ v2→v3 |
| :----------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| word_count         |     5764.8 |     9295.1 |    +3530.3 |    10184.2 |     +889.1 |
| citation_count     |       48.7 |       65.5 |      +16.7 |       75.6 |      +10.2 |
| section_count      |       38.5 |       59.8 |      +21.3 |       63.3 |       +3.5 |