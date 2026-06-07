# gpt4.1 — Domain-Level Breakdown


## Turns (v1 / v2 / v3)

### Turns — Input / Output Tokens (mean per task)

| Domain                         |  InTok gpt41_turn1 | OutTok gpt41_turn1 |  InTok gpt41_turn2 | OutTok gpt41_turn2 |  InTok gpt41_turn3 | OutTok gpt41_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |      886,425 (n=6) |             57,087 |    1,040,534 (n=6) |             64,620 |      881,272 (n=6) |             54,653 |
| Finance                        |     900,383 (n=10) |             50,351 |   1,453,804 (n=10) |             78,930 |   1,089,000 (n=10) |             58,261 |
| General Knowledge              |      774,666 (n=5) |             55,069 |    1,191,940 (n=5) |             74,913 |    1,040,984 (n=5) |             67,947 |
| Law                            |      199,744 (n=3) |             18,165 |      149,262 (n=3) |             16,083 |      410,015 (n=3) |             27,061 |
| Medicine                       |    1,180,856 (n=3) |             65,016 |      801,563 (n=3) |             48,179 |      606,651 (n=3) |             41,235 |
| Needle in a Haystack           |      206,651 (n=3) |             16,462 |    1,003,682 (n=3) |             60,780 |      654,594 (n=3) |             39,065 |
| Personalized Assistant         |      538,069 (n=3) |             45,614 |      800,007 (n=3) |             59,889 |    1,452,056 (n=3) |            103,548 |
| Shopping/Product Comparison    |    1,110,403 (n=8) |             73,020 |    1,191,746 (n=8) |             78,471 |    1,030,626 (n=8) |             63,967 |
| Technology                     |      681,814 (n=5) |             47,133 |    1,181,591 (n=5) |             77,346 |    1,146,218 (n=5) |             79,168 |
| UX Design                      |      839,484 (n=4) |             49,139 |    1,029,908 (n=4) |             60,428 |    1,123,187 (n=4) |             65,181 |

### Turns — Cost USD (mean per task)

| Domain                         | Cost gpt41_turn1 | Cost gpt41_turn2 | Cost gpt41_turn3 |
| :----------------------------- | ---------------: | ---------------: | ---------------: |
| Academic                       |          $0.8154 |          $1.0017 |          $0.7959 |
| Finance                        |          $0.7397 |          $1.2234 |          $0.8355 |
| General Knowledge              |          $0.7505 |          $1.0906 |          $0.9512 |
| Law                            |          $0.2370 |          $0.2161 |          $0.3745 |
| Medicine                       |          $0.9898 |          $0.7397 |          $0.5772 |
| Needle in a Haystack           |          $0.2216 |          $0.8741 |          $0.5708 |
| Personalized Assistant         |          $0.5718 |          $0.8140 |          $1.4028 |
| Shopping/Product Comparison    |          $1.0392 |          $1.1296 |          $0.9237 |
| Technology                     |          $0.6506 |          $1.0786 |          $1.0808 |
| UX Design                      |          $0.7166 |          $0.9133 |          $0.9722 |

### Turns — Latency (mean seconds per task)

| Domain                         | Latency gpt41_turn1 | Latency gpt41_turn2 | Latency gpt41_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: |
| Academic                       |             185.7s |             251.3s |             226.8s |
| Finance                        |             252.4s |             234.0s |             435.9s |
| General Knowledge              |             183.6s |             230.1s |             272.8s |
| Law                            |             126.8s |             107.0s |             186.9s |
| Medicine                       |             270.3s |             160.4s |             314.9s |
| Needle in a Haystack           |             113.2s |             220.6s |             261.9s |
| Personalized Assistant         |             241.7s |             219.7s |             290.4s |
| Shopping/Product Comparison    |             230.2s |             282.9s |             267.0s |
| Technology                     |             176.5s |             243.7s |             247.7s |
| UX Design                      |             226.0s |             235.0s |             245.3s |

### Turns — Researchers Spawned (mean per task)

| Domain                         | Researchers gpt41_turn1 | Researchers gpt41_turn2 | Researchers gpt41_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    3.7 |                    4.0 |                    3.5 |
| Finance                        |                    2.8 |                    3.9 |                    2.9 |
| General Knowledge              |                    2.6 |                    3.2 |                    2.8 |
| Law                            |                    1.0 |                    1.0 |                    1.0 |
| Medicine                       |                    3.7 |                    2.7 |                    2.3 |
| Needle in a Haystack           |                    1.3 |                    3.7 |                    2.0 |
| Personalized Assistant         |                    2.3 |                    2.7 |                    5.3 |
| Shopping/Product Comparison    |                    3.4 |                    4.2 |                    2.9 |
| Technology                     |                    2.6 |                    4.0 |                    3.6 |
| UX Design                      |                    2.8 |                    3.0 |                    3.2 |

### Turns — Search Calls & URLs (mean per task)

| Domain                         |   Searches gpt41_turn1 |       URLs gpt41_turn1 |   Searches gpt41_turn2 |       URLs gpt41_turn2 |   Searches gpt41_turn3 |       URLs gpt41_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    7.0 |                  108.0 |                    7.5 |                  128.0 |                    7.3 |                  106.3 |
| Finance                        |                    7.2 |                   82.5 |                   12.6 |                  120.4 |                   10.4 |                   80.3 |
| General Knowledge              |                    4.4 |                   93.2 |                    9.6 |                  129.6 |                    7.2 |                  119.8 |
| Law                            |                    2.0 |                   29.3 |                    1.0 |                   19.7 |                    3.0 |                   39.7 |
| Medicine                       |                    8.0 |                  111.3 |                    6.0 |                   68.7 |                    4.7 |                   70.0 |
| Needle in a Haystack           |                    2.3 |                   35.7 |                    9.7 |                  136.3 |                    5.3 |                   74.0 |
| Personalized Assistant         |                    4.7 |                   85.7 |                    5.0 |                  111.7 |                   16.7 |                  179.3 |
| Shopping/Product Comparison    |                    7.2 |                  152.5 |                    9.4 |                  154.0 |                    7.6 |                  137.5 |
| Technology                     |                    4.8 |                   84.8 |                    7.8 |                  134.4 |                    6.4 |                  144.6 |
| UX Design                      |                    5.8 |                  112.8 |                    5.8 |                  131.2 |                    7.8 |                  150.0 |

## T1 vs Self-Reflect

### Self-Reflect — Input / Output Tokens (mean per task)

| Domain                         |  InTok gpt41_turn1 | OutTok gpt41_turn1 | InTok gpt41_self_reflect | OutTok gpt41_self_reflect |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |      886,425 (n=6) |             57,087 |      969,914 (n=6) |             63,710 |
| Finance                        |     900,383 (n=10) |             50,351 |   1,198,512 (n=10) |             66,154 |
| General Knowledge              |      774,666 (n=5) |             55,069 |      980,682 (n=5) |             62,856 |
| Law                            |      199,744 (n=3) |             18,165 |      469,211 (n=3) |             36,331 |
| Medicine                       |    1,180,856 (n=3) |             65,016 |      597,049 (n=3) |             39,369 |
| Needle in a Haystack           |      206,651 (n=3) |             16,462 |      393,347 (n=3) |             29,915 |
| Personalized Assistant         |      538,069 (n=3) |             45,614 |    1,154,029 (n=3) |             83,826 |
| Shopping/Product Comparison    |    1,110,403 (n=8) |             73,020 |    1,096,646 (n=8) |             73,710 |
| Technology                     |      681,814 (n=5) |             47,133 |    1,039,443 (n=5) |             70,497 |
| UX Design                      |      839,484 (n=4) |             49,139 |      648,365 (n=4) |             40,080 |

### Self-Reflect — Cost USD (mean per task)

| Domain                         | Cost gpt41_turn1 | Cost gpt41_self_reflect |
| :----------------------------- | ---------------: | ---------------: |
| Academic                       |          $0.8154 |          $0.8942 |
| Finance                        |          $0.7397 |          $0.9580 |
| General Knowledge              |          $0.7505 |          $0.8664 |
| Law                            |          $0.2370 |          $0.4693 |
| Medicine                       |          $0.9898 |          $0.5501 |
| Needle in a Haystack           |          $0.2216 |          $0.4010 |
| Personalized Assistant         |          $0.5718 |          $1.0816 |
| Shopping/Product Comparison    |          $1.0392 |          $1.0041 |
| Technology                     |          $0.6506 |          $0.9500 |
| UX Design                      |          $0.7166 |          $0.5686 |

### Self-Reflect — Latency (mean seconds per task)

| Domain                         | Latency gpt41_turn1 | Latency gpt41_self_reflect |
| :----------------------------- | -----------------: | -----------------: |
| Academic                       |             185.7s |             164.5s |
| Finance                        |             252.4s |             290.6s |
| General Knowledge              |             183.6s |             206.9s |
| Law                            |             126.8s |             149.3s |
| Medicine                       |             270.3s |             213.9s |
| Needle in a Haystack           |             113.2s |             164.2s |
| Personalized Assistant         |             241.7s |             287.4s |
| Shopping/Product Comparison    |             230.2s |             228.8s |
| Technology                     |             176.5s |             205.6s |
| UX Design                      |             226.0s |             200.3s |

### Self-Reflect — Researchers Spawned (mean per task)

| Domain                         | Researchers gpt41_turn1 | Researchers gpt41_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: |
| Academic                       |                    3.7 |                    4.2 |
| Finance                        |                    2.8 |                    3.5 |
| General Knowledge              |                    2.6 |                    2.8 |
| Law                            |                    1.0 |                    2.3 |
| Medicine                       |                    3.7 |                    2.7 |
| Needle in a Haystack           |                    1.3 |                    2.3 |
| Personalized Assistant         |                    2.3 |                    4.0 |
| Shopping/Product Comparison    |                    3.4 |                    3.6 |
| Technology                     |                    2.6 |                    3.8 |
| UX Design                      |                    2.8 |                    2.2 |

### Self-Reflect — Search Calls & URLs (mean per task)

| Domain                         |   Searches gpt41_turn1 |       URLs gpt41_turn1 | Searches gpt41_self_reflect | URLs gpt41_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    7.0 |                  108.0 |                    7.8 |                  125.7 |
| Finance                        |                    7.2 |                   82.5 |                   10.2 |                  155.3 |
| General Knowledge              |                    4.4 |                   93.2 |                    5.8 |                  110.4 |
| Law                            |                    2.0 |                   29.3 |                    3.7 |                   54.3 |
| Medicine                       |                    8.0 |                  111.3 |                    5.3 |                   65.0 |
| Needle in a Haystack           |                    2.3 |                   35.7 |                    4.7 |                   55.7 |
| Personalized Assistant         |                    4.7 |                   85.7 |                    8.7 |                  144.7 |
| Shopping/Product Comparison    |                    7.2 |                  152.5 |                    8.6 |                  149.6 |
| Technology                     |                    4.8 |                   84.8 |                    6.2 |                  137.8 |
| UX Design                      |                    5.8 |                  112.8 |                    4.2 |                   80.0 |