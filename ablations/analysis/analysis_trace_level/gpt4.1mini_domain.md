# gpt4.1mini — Domain-Level Breakdown


## Turns (v1 / v2 / v3)

### Turns — Input / Output Tokens (mean per task)

| Domain                         |   InTok mini_turn1 |  OutTok mini_turn1 |   InTok mini_turn2 |  OutTok mini_turn2 |   InTok mini_turn3 |  OutTok mini_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |      890,733 (n=6) |             55,804 |      713,498 (n=6) |             49,139 |      521,360 (n=6) |             38,444 |
| Finance                        |     977,603 (n=10) |             51,426 |   1,305,175 (n=10) |             69,893 |   1,757,204 (n=10) |             87,555 |
| General Knowledge              |    1,171,470 (n=5) |             71,617 |    1,494,478 (n=5) |             96,424 |    1,636,115 (n=5) |            103,554 |
| Law                            |      170,146 (n=3) |             16,235 |      804,080 (n=3) |             51,458 |      804,699 (n=3) |             52,153 |
| Medicine                       |      856,114 (n=3) |             47,612 |      403,676 (n=3) |             28,422 |      740,186 (n=3) |             47,447 |
| Needle in a Haystack           |      246,880 (n=3) |             19,812 |      566,659 (n=3) |             40,271 |      759,069 (n=3) |             47,560 |
| Personalized Assistant         |      812,137 (n=3) |             61,875 |    1,191,412 (n=3) |             77,225 |    1,619,627 (n=3) |            116,841 |
| Shopping/Product Comparison    |      874,272 (n=8) |             60,847 |    1,583,311 (n=8) |             94,152 |    1,862,233 (n=8) |            114,682 |
| Technology                     |    1,015,331 (n=5) |             68,668 |    1,364,041 (n=5) |             88,716 |    1,199,960 (n=5) |             83,922 |
| UX Design                      |    1,530,741 (n=4) |             86,682 |    1,497,277 (n=4) |             85,016 |    1,176,631 (n=4) |             67,352 |

### Turns — Cost USD (mean per task)

| Domain                         |  Cost mini_turn1 |  Cost mini_turn2 |  Cost mini_turn3 |
| :----------------------------- | ---------------: | ---------------: | ---------------: |
| Academic                       |          $0.3989 |          $0.3290 |          $0.2575 |
| Finance                        |          $0.3953 |          $0.5138 |          $0.6540 |
| General Knowledge              |          $0.5202 |          $0.6880 |          $0.7453 |
| Law                            |          $0.0866 |          $0.3462 |          $0.3480 |
| Medicine                       |          $0.3735 |          $0.1920 |          $0.3093 |
| Needle in a Haystack           |          $0.1179 |          $0.2533 |          $0.3271 |
| Personalized Assistant         |          $0.3854 |          $0.5203 |          $0.7746 |
| Shopping/Product Comparison    |          $0.3985 |          $0.6706 |          $0.8094 |
| Technology                     |          $0.4659 |          $0.6330 |          $0.5686 |
| UX Design                      |          $0.6585 |          $0.6507 |          $0.5230 |

### Turns — Latency (mean seconds per task)

| Domain                         | Latency mini_turn1 | Latency mini_turn2 | Latency mini_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: |
| Academic                       |             318.7s |             187.7s |             212.4s |
| Finance                        |             227.8s |             223.5s |             356.8s |
| General Knowledge              |             209.9s |             251.0s |             311.2s |
| Law                            |             137.9s |             111.5s |             148.1s |
| Medicine                       |             351.9s |             106.2s |             358.6s |
| Needle in a Haystack           |             130.7s |             186.9s |             310.4s |
| Personalized Assistant         |             202.1s |             170.8s |             341.4s |
| Shopping/Product Comparison    |             213.1s |             261.1s |             401.2s |
| Technology                     |             253.5s |             184.4s |             291.0s |
| UX Design                      |             376.5s |             279.6s |             357.9s |

### Turns — Researchers Spawned (mean per task)

| Domain                         | Researchers mini_turn1 | Researchers mini_turn2 | Researchers mini_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    3.8 |                    2.5 |                    2.8 |
| Finance                        |                    1.8 |                    2.8 |                    3.0 |
| General Knowledge              |                    2.8 |                    4.2 |                    3.2 |
| Law                            |                    1.0 |                    2.3 |                    2.3 |
| Medicine                       |                    2.3 |                    1.0 |                    1.7 |
| Needle in a Haystack           |                    2.0 |                    2.0 |                    2.7 |
| Personalized Assistant         |                    2.3 |                    2.3 |                    4.7 |
| Shopping/Product Comparison    |                    2.2 |                    2.6 |                    3.2 |
| Technology                     |                    2.8 |                    3.0 |                    3.6 |
| UX Design                      |                    3.2 |                    3.2 |                    2.2 |

### Turns — Search Calls & URLs (mean per task)

| Domain                         |    Searches mini_turn1 |        URLs mini_turn1 |    Searches mini_turn2 |        URLs mini_turn2 |    Searches mini_turn3 |        URLs mini_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    6.2 |                  101.3 |                    3.2 |                   95.0 |                    2.8 |                   74.0 |
| Finance                        |                    5.8 |                   82.5 |                   10.7 |                  110.6 |                   13.8 |                  118.7 |
| General Knowledge              |                    6.0 |                  127.4 |                    8.4 |                  186.2 |                    6.8 |                  181.2 |
| Law                            |                    1.3 |                   25.7 |                    4.7 |                   75.3 |                    4.7 |                   75.3 |
| Medicine                       |                    4.7 |                   88.0 |                    2.3 |                   49.3 |                    3.3 |                   76.7 |
| Needle in a Haystack           |                    3.0 |                   43.3 |                    5.3 |                   87.7 |                    7.7 |                   97.3 |
| Personalized Assistant         |                    5.7 |                  119.0 |                    8.0 |                  155.7 |                   10.0 |                  251.7 |
| Shopping/Product Comparison    |                    5.9 |                  131.0 |                    7.1 |                  199.0 |                    9.2 |                  237.5 |
| Technology                     |                    5.0 |                  144.8 |                    6.0 |                  178.8 |                    7.6 |                  163.8 |
| UX Design                      |                    8.0 |                  213.2 |                    9.0 |                  189.8 |                    6.2 |                  156.0 |

## T1 vs Self-Reflect

### Self-Reflect — Input / Output Tokens (mean per task)

| Domain                         |   InTok mini_turn1 |  OutTok mini_turn1 | InTok mini_self_reflect | OutTok mini_self_reflect |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |      890,733 (n=6) |             55,804 |      690,379 (n=6) |             46,367 |
| Finance                        |     977,603 (n=10) |             51,426 |     750,174 (n=10) |             45,960 |
| General Knowledge              |    1,171,470 (n=5) |             71,617 |    1,110,366 (n=5) |             71,860 |
| Law                            |      170,146 (n=3) |             16,235 |      219,322 (n=3) |             19,951 |
| Medicine                       |      856,114 (n=3) |             47,612 |      741,313 (n=3) |             46,186 |
| Needle in a Haystack           |      246,880 (n=3) |             19,812 |      396,278 (n=3) |             25,263 |
| Personalized Assistant         |      812,137 (n=3) |             61,875 |    1,129,084 (n=3) |             76,132 |
| Shopping/Product Comparison    |      874,272 (n=8) |             60,847 |      998,120 (n=8) |             65,277 |
| Technology                     |    1,015,331 (n=5) |             68,668 |      977,311 (n=5) |             74,719 |
| UX Design                      |    1,530,741 (n=4) |             86,682 |    1,976,962 (n=4) |            104,498 |

### Self-Reflect — Cost USD (mean per task)

| Domain                         |  Cost mini_turn1 | Cost mini_self_reflect |
| :----------------------------- | ---------------: | ---------------: |
| Academic                       |          $0.3989 |          $0.3231 |
| Finance                        |          $0.3953 |          $0.3332 |
| General Knowledge              |          $0.5202 |          $0.5188 |
| Law                            |          $0.0866 |          $0.1159 |
| Medicine                       |          $0.3735 |          $0.3414 |
| Needle in a Haystack           |          $0.1179 |          $0.1780 |
| Personalized Assistant         |          $0.3854 |          $0.5216 |
| Shopping/Product Comparison    |          $0.3985 |          $0.4568 |
| Technology                     |          $0.4659 |          $0.4740 |
| UX Design                      |          $0.6585 |          $0.8413 |

### Self-Reflect — Latency (mean seconds per task)

| Domain                         | Latency mini_turn1 | Latency mini_self_reflect |
| :----------------------------- | -----------------: | -----------------: |
| Academic                       |             318.7s |             229.6s |
| Finance                        |             227.8s |             243.7s |
| General Knowledge              |             209.9s |             225.5s |
| Law                            |             137.9s |             138.3s |
| Medicine                       |             351.9s |             154.1s |
| Needle in a Haystack           |             130.7s |             204.9s |
| Personalized Assistant         |             202.1s |             330.3s |
| Shopping/Product Comparison    |             213.1s |             304.2s |
| Technology                     |             253.5s |             271.8s |
| UX Design                      |             376.5s |             315.7s |

### Self-Reflect — Researchers Spawned (mean per task)

| Domain                         | Researchers mini_turn1 | Researchers mini_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: |
| Academic                       |                    3.8 |                    2.5 |
| Finance                        |                    1.8 |                    1.8 |
| General Knowledge              |                    2.8 |                    2.8 |
| Law                            |                    1.0 |                    1.0 |
| Medicine                       |                    2.3 |                    3.0 |
| Needle in a Haystack           |                    2.0 |                    1.3 |
| Personalized Assistant         |                    2.3 |                    2.7 |
| Shopping/Product Comparison    |                    2.2 |                    2.0 |
| Technology                     |                    2.8 |                    3.8 |
| UX Design                      |                    3.2 |                    3.5 |

### Self-Reflect — Search Calls & URLs (mean per task)

| Domain                         |    Searches mini_turn1 |        URLs mini_turn1 | Searches mini_self_reflect | URLs mini_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    6.2 |                  101.3 |                    4.0 |                   85.8 |
| Finance                        |                    5.8 |                   82.5 |                    5.2 |                   77.3 |
| General Knowledge              |                    6.0 |                  127.4 |                    4.6 |                  144.0 |
| Law                            |                    1.3 |                   25.7 |                    1.0 |                   31.0 |
| Medicine                       |                    4.7 |                   88.0 |                    4.7 |                   72.7 |
| Needle in a Haystack           |                    3.0 |                   43.3 |                    3.3 |                   62.3 |
| Personalized Assistant         |                    5.7 |                  119.0 |                    7.0 |                  144.7 |
| Shopping/Product Comparison    |                    5.9 |                  131.0 |                    5.8 |                  152.9 |
| Technology                     |                    5.0 |                  144.8 |                    5.6 |                  138.0 |
| UX Design                      |                    8.0 |                  213.2 |                   10.8 |                  249.8 |