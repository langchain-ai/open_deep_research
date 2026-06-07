# gpt4.1mini — T1 vs Self-Reflect Comparison


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| mini_turn1             |   50 |        908,548 |         56,412 |   $19.9884 |       243.7s |
| mini_self_reflect      |   50 |        908,664 |         58,270 |   $20.7255 |       249.6s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| mini_turn1             |    2.5 / 3.0 / 6 | 11.8 / 12.0 / 30 | 5.5 / 6.0 / 21 | 110.5 / 96.5 / 294 |
| mini_self_reflect      |    2.4 / 2.0 / 6 | 11.7 / 10.0 / 32 | 5.3 / 5.0 / 13 | 117.0 / 91.0 / 296 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| mini_turn1             |          $0.3030 (75.8%) |         $0.0867 (21.68%) |          $0.0006 (0.15%) |          $0.0095 (2.37%) |
| mini_self_reflect      |         $0.3064 (72.45%) |         $0.1004 (24.23%) |           $0.0021 (0.5%) |          $0.0117 (2.82%) |

## Report Characteristics — T1 vs Self-Reflect v2

| Metric             |      T1 (v1) |  Self-Reflect v2 |    Δ (SR − T1) |
| :----------------- | -----------: | ---------------: | -------------: |
| word_count         |       2052.2 |           2264.3 |         +212.1 |
| citation_count     |         19.6 |             20.9 |           +1.3 |
| section_count      |         21.7 |             24.2 |           +2.5 |