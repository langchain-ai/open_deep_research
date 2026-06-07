# deepseekv4flash — T1 vs Self-Reflect Comparison


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| deepseek_turn1         |   50 |      2,557,090 |        156,617 |   $35.2076 |       459.6s |
| deepseek_self_reflect  |   50 |      3,695,454 |        236,413 |   $54.2376 |       538.7s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| deepseek_turn1         |    4.2 / 4.0 / 8 | 36.2 / 35.0 / 80 | 23.4 / 21.5 / 73 | 262.9 / 279.5 / 466 |
| deepseek_self_reflect  |   6.2 / 6.0 / 11 | 50.6 / 50.0 / 110 | 32.3 / 32.0 / 87 | 378.6 / 383.0 / 707 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| deepseek_turn1         |         $0.7042 (100.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |
| deepseek_self_reflect  |         $1.0848 (100.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |           $0.0000 (0.0%) |

## Report Characteristics — T1 vs Self-Reflect v2

| Metric             |      T1 (v1) |  Self-Reflect v2 |    Δ (SR − T1) |
| :----------------- | -----------: | ---------------: | -------------: |
| word_count         |       5764.8 |           8181.8 |        +2417.0 |
| citation_count     |         48.7 |             66.8 |          +18.1 |
| section_count      |         38.5 |             54.5 |          +16.0 |