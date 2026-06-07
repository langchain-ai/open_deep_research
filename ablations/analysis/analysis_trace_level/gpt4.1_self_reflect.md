# gpt4.1 — T1 vs Self-Reflect Comparison


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| gpt41_turn1            |   50 |        804,438 |         51,471 |   $36.5359 |       208.8s |
| gpt41_self_reflect     |   50 |        942,255 |         60,578 |   $41.8404 |       220.6s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| gpt41_turn1            |    2.8 / 3.0 / 6 | 14.3 / 13.5 / 32 | 5.8 / 6.0 / 13 | 96.4 / 96.0 / 210 |
| gpt41_self_reflect     |    3.3 / 3.0 / 6 | 17.0 / 15.0 / 36 | 7.2 / 6.5 / 23 | 120.5 / 99.0 / 570 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| gpt41_turn1            |          $0.2689 (36.8%) |         $0.4052 (55.45%) |          $0.0033 (0.45%) |           $0.0533 (7.3%) |
| gpt41_self_reflect     |         $0.2959 (35.36%) |         $0.4615 (55.15%) |          $0.0122 (1.46%) |          $0.0672 (8.04%) |

## Report Characteristics — T1 vs Self-Reflect v2

| Metric             |      T1 (v1) |  Self-Reflect v2 |    Δ (SR − T1) |
| :----------------- | -----------: | ---------------: | -------------: |
| word_count         |       2140.5 |           2267.3 |         +126.8 |
| citation_count     |         28.7 |             29.8 |           +1.1 |
| section_count      |         22.8 |             24.1 |           +1.3 |