# gpt4.1 — Turns Analysis (v1 / v2 / v3)


## Overall — Token / Cost / Latency
> mean per task; Total $ is summed across all tasks in the project

| Label                  |    n |   InTok (mean) |  OutTok (mean) |    Total $ |  Avg Lat (s) |
| :--------------------- | ---: | -------------: | -------------: | ---------: | -----------: |
| gpt41_turn1            |   50 |        804,438 |         51,471 |   $36.5359 |       208.8s |
| gpt41_turn2            |   50 |      1,091,321 |         67,252 |   $49.7110 |       230.9s |
| gpt41_turn3            |   50 |        984,427 |         61,026 |   $43.3452 |       289.1s |

## Overall — Count Metrics
> format: mean / median / max per task

| Label                  |      Researchers |    React iters |       Searches |           URLs |
| :--------------------- | ---------------: | -------------: | -------------: | -------------: |
| gpt41_turn1            |    2.8 / 3.0 / 6 | 14.3 / 13.5 / 32 | 5.8 / 6.0 / 13 | 96.4 / 96.0 / 210 |
| gpt41_turn2            |    3.5 / 3.0 / 9 | 18.9 / 19.5 / 47 | 8.4 / 8.0 / 23 | 121.2 / 116.0 / 342 |
| gpt41_turn3            |    3.0 / 3.0 / 7 | 17.3 / 17.0 / 45 | 7.9 / 7.5 / 27 | 111.0 / 97.5 / 265 |

## Overall — Phase Cost Breakdown
> mean $ per task | % of project total cost

| Label                  |                    other |                 research |                  scoping |                  writing |
| :--------------------- | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| gpt41_turn1            |          $0.2689 (36.8%) |         $0.4052 (55.45%) |          $0.0033 (0.45%) |           $0.0533 (7.3%) |
| gpt41_turn2            |         $0.3554 (35.03%) |         $0.5566 (55.99%) |          $0.0135 (1.36%) |          $0.0758 (7.62%) |
| gpt41_turn3            |         $0.3038 (35.75%) |         $0.4782 (56.27%) |          $0.0152 (1.79%) |          $0.0745 (8.59%) |

## Report Characteristics
> mean per task across all tasks with that version

| Metric             |    v1 mean |    v2 mean |    Δ v1→v2 |    v3 mean |    Δ v2→v3 |
| :----------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| word_count         |     2140.5 |     2550.0 |     +409.5 |     2510.0 |      -40.0 |
| citation_count     |       28.7 |       32.4 |       +3.7 |       30.1 |       -2.4 |
| section_count      |       22.8 |       26.0 |       +3.2 |       26.3 |       +0.3 |