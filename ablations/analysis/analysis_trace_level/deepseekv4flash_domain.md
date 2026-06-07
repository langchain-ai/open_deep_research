# deepseekv4flash — Domain-Level Breakdown


## Turns (v1 / v2 / v3)

### Turns — Input / Output Tokens (mean per task)

| Domain                         | InTok deepseek_turn1 | OutTok deepseek_turn1 | InTok deepseek_turn2 | OutTok deepseek_turn2 | InTok deepseek_turn3 | OutTok deepseek_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |    2,816,056 (n=6) |            169,192 |    2,571,161 (n=6) |            163,697 |    3,425,311 (n=5) |            208,219 |
| Finance                        |   3,164,522 (n=10) |            165,587 |   5,389,954 (n=10) |            270,755 |   4,396,485 (n=10) |            236,262 |
| General Knowledge              |    3,239,366 (n=5) |            202,748 |    3,602,074 (n=5) |            187,201 |    4,390,531 (n=5) |            271,092 |
| Law                            |    1,538,933 (n=3) |             93,103 |    2,685,819 (n=3) |            164,028 |    3,570,997 (n=2) |            235,800 |
| Medicine                       |    2,284,409 (n=3) |            140,296 |    2,773,676 (n=3) |            166,081 |    2,531,330 (n=3) |            162,535 |
| Needle in a Haystack           |    1,031,055 (n=3) |             56,038 |    2,991,448 (n=3) |            186,456 |    2,173,869 (n=2) |            132,918 |
| Personalized Assistant         |    1,823,039 (n=3) |            141,287 |    2,611,034 (n=3) |            174,835 |    3,757,819 (n=3) |            271,273 |
| Shopping/Product Comparison    |    2,792,748 (n=8) |            187,964 |    4,722,235 (n=8) |            283,801 |    4,801,288 (n=8) |            310,063 |
| Technology                     |    2,198,887 (n=5) |            146,306 |    4,333,204 (n=5) |            239,060 |    4,901,177 (n=5) |            304,582 |
| UX Design                      |    2,436,846 (n=4) |            154,664 |    3,191,184 (n=4) |            178,946 |    3,401,211 (n=4) |            218,281 |

### Turns — Cost USD (mean per task)

| Domain                         | Cost deepseek_turn1 | Cost deepseek_turn2 | Cost deepseek_turn3 |
| :----------------------------- | ---------------: | ---------------: | ---------------: |
| Academic                       |          $0.7923 |          $0.7390 |          $0.9759 |
| Finance                        |          $0.8110 |          $1.2500 |          $1.1182 |
| General Knowledge              |          $0.8936 |          $0.7766 |          $1.1753 |
| Law                            |          $0.3669 |          $0.5652 |          $0.8642 |
| Medicine                       |          $0.6451 |          $0.6781 |          $0.7039 |
| Needle in a Haystack           |          $0.2440 |          $0.7814 |          $0.4762 |
| Personalized Assistant         |          $0.4916 |          $0.6757 |          $0.9768 |
| Shopping/Product Comparison    |          $0.7688 |          $1.1205 |          $1.2930 |
| Technology                     |          $0.7018 |          $1.0596 |          $1.4007 |
| UX Design                      |          $0.7434 |          $0.7664 |          $0.9928 |

### Turns — Latency (mean seconds per task)

| Domain                         | Latency deepseek_turn1 | Latency deepseek_turn2 | Latency deepseek_turn3 |
| :----------------------------- | -----------------: | -----------------: | -----------------: |
| Academic                       |             470.3s |             548.3s |             608.2s |
| Finance                        |             542.1s |             875.6s |             716.6s |
| General Knowledge              |             505.7s |             931.3s |             659.8s |
| Law                            |             407.2s |             738.4s |             571.0s |
| Medicine                       |             401.6s |             581.0s |             520.2s |
| Needle in a Haystack           |             283.0s |             631.6s |             659.2s |
| Personalized Assistant         |             418.9s |             720.3s |             649.2s |
| Shopping/Product Comparison    |             552.8s |             930.0s |             775.4s |
| Technology                     |             341.2s |             916.6s |             780.9s |
| UX Design                      |             387.2s |             677.3s |             630.6s |

### Turns — Researchers Spawned (mean per task)

| Domain                         | Researchers deepseek_turn1 | Researchers deepseek_turn2 | Researchers deepseek_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                    4.8 |                    4.3 |                    5.2 |
| Finance                        |                    4.3 |                    6.5 |                    5.8 |
| General Knowledge              |                    5.2 |                    5.0 |                    6.2 |
| Law                            |                    2.7 |                    4.3 |                    6.0 |
| Medicine                       |                    4.0 |                    4.7 |                    4.3 |
| Needle in a Haystack           |                    2.3 |                    5.7 |                    4.0 |
| Personalized Assistant         |                    3.7 |                    4.3 |                    7.3 |
| Shopping/Product Comparison    |                    5.0 |                    7.4 |                    7.1 |
| Technology                     |                    3.6 |                    6.0 |                    6.8 |
| UX Design                      |                    4.5 |                    5.5 |                    5.8 |

### Turns — Search Calls & URLs (mean per task)

| Domain                         | Searches deepseek_turn1 |    URLs deepseek_turn1 | Searches deepseek_turn2 |    URLs deepseek_turn2 | Searches deepseek_turn3 |    URLs deepseek_turn3 |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                   24.7 |                  298.0 |                   21.5 |                  301.7 |                   26.6 |                  343.8 |
| Finance                        |                   30.7 |                  235.5 |                   49.1 |                  490.1 |                   42.2 |                  288.3 |
| General Knowledge              |                   26.4 |                  330.0 |                   23.6 |                  793.4 |                   34.6 |                  350.8 |
| Law                            |                   13.0 |                  191.3 |                   20.7 |                  225.7 |                   31.0 |                  206.5 |
| Medicine                       |                   20.3 |                  195.0 |                   19.3 |                  397.7 |                   25.3 |                  215.7 |
| Needle in a Haystack           |                   12.0 |                  129.0 |                   33.7 |                  348.7 |                   23.0 |                  257.5 |
| Personalized Assistant         |                   15.7 |                  189.7 |                   23.0 |                  394.3 |                   32.0 |                  426.7 |
| Shopping/Product Comparison    |                   28.0 |                  329.4 |                   39.9 |                 1018.5 |                   44.1 |                  490.2 |
| Technology                     |                   17.0 |                  275.8 |                   30.2 |                 1002.4 |                   37.4 |                  513.2 |
| UX Design                      |                   23.2 |                  306.2 |                   23.0 |                  899.2 |                   30.5 |                  414.0 |

## T1 vs Self-Reflect

### Self-Reflect — Input / Output Tokens (mean per task)

| Domain                         | InTok deepseek_turn1 | OutTok deepseek_turn1 | InTok deepseek_self_reflect | OutTok deepseek_self_reflect |
| :----------------------------- | -----------------: | -----------------: | -----------------: | -----------------: |
| Academic                       |    2,816,056 (n=6) |            169,192 |    3,896,570 (n=6) |            241,082 |
| Finance                        |   3,164,522 (n=10) |            165,587 |   4,272,935 (n=10) |            235,888 |
| General Knowledge              |    3,239,366 (n=5) |            202,748 |    4,262,791 (n=5) |            281,854 |
| Law                            |    1,538,933 (n=3) |             93,103 |    1,541,327 (n=3) |            115,450 |
| Medicine                       |    2,284,409 (n=3) |            140,296 |    2,891,142 (n=3) |            202,669 |
| Needle in a Haystack           |    1,031,055 (n=3) |             56,038 |    2,756,164 (n=3) |            149,600 |
| Personalized Assistant         |    1,823,039 (n=3) |            141,287 |    3,340,222 (n=3) |            253,482 |
| Shopping/Product Comparison    |    2,792,748 (n=8) |            187,964 |    4,287,532 (n=8) |            290,766 |
| Technology                     |    2,198,887 (n=5) |            146,306 |    3,167,803 (n=5) |            225,244 |
| UX Design                      |    2,436,846 (n=4) |            154,664 |    3,906,031 (n=4) |            247,515 |

### Self-Reflect — Cost USD (mean per task)

| Domain                         | Cost deepseek_turn1 | Cost deepseek_self_reflect |
| :----------------------------- | ---------------: | ---------------: |
| Academic                       |          $0.7923 |          $1.1693 |
| Finance                        |          $0.8110 |          $1.2104 |
| General Knowledge              |          $0.8936 |          $1.2663 |
| Law                            |          $0.3669 |          $0.4421 |
| Medicine                       |          $0.6451 |          $0.8361 |
| Needle in a Haystack           |          $0.2440 |          $0.7344 |
| Personalized Assistant         |          $0.4916 |          $0.9641 |
| Shopping/Product Comparison    |          $0.7688 |          $1.2701 |
| Technology                     |          $0.7018 |          $0.9668 |
| UX Design                      |          $0.7434 |          $1.2155 |

### Self-Reflect — Latency (mean seconds per task)

| Domain                         | Latency deepseek_turn1 | Latency deepseek_self_reflect |
| :----------------------------- | -----------------: | -----------------: |
| Academic                       |             470.3s |             547.7s |
| Finance                        |             542.1s |             607.7s |
| General Knowledge              |             505.7s |             585.1s |
| Law                            |             407.2s |             338.0s |
| Medicine                       |             401.6s |             485.7s |
| Needle in a Haystack           |             283.0s |             401.5s |
| Personalized Assistant         |             418.9s |             540.6s |
| Shopping/Product Comparison    |             552.8s |             579.1s |
| Technology                     |             341.2s |             453.7s |
| UX Design                      |             387.2s |             611.9s |

### Self-Reflect — Researchers Spawned (mean per task)

| Domain                         | Researchers deepseek_turn1 | Researchers deepseek_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: |
| Academic                       |                    4.8 |                    6.0 |
| Finance                        |                    4.3 |                    5.8 |
| General Knowledge              |                    5.2 |                    7.0 |
| Law                            |                    2.7 |                    3.3 |
| Medicine                       |                    4.0 |                    6.0 |
| Needle in a Haystack           |                    2.3 |                    5.0 |
| Personalized Assistant         |                    3.7 |                    6.3 |
| Shopping/Product Comparison    |                    5.0 |                    7.6 |
| Technology                     |                    3.6 |                    5.8 |
| UX Design                      |                    4.5 |                    7.0 |

### Self-Reflect — Search Calls & URLs (mean per task)

| Domain                         | Searches deepseek_turn1 |    URLs deepseek_turn1 | Searches deepseek_self_reflect | URLs deepseek_self_reflect |
| :----------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| Academic                       |                   24.7 |                  298.0 |                   33.0 |                  445.0 |
| Finance                        |                   30.7 |                  235.5 |                   37.2 |                  306.5 |
| General Knowledge              |                   26.4 |                  330.0 |                   35.0 |                  451.8 |
| Law                            |                   13.0 |                  191.3 |                   16.3 |                  145.7 |
| Medicine                       |                   20.3 |                  195.0 |                   23.0 |                  307.3 |
| Needle in a Haystack           |                   12.0 |                  129.0 |                   28.3 |                  279.0 |
| Personalized Assistant         |                   15.7 |                  189.7 |                   28.3 |                  349.0 |
| Shopping/Product Comparison    |                   28.0 |                  329.4 |                   38.9 |                  474.0 |
| Technology                     |                   17.0 |                  275.8 |                   28.0 |                  385.6 |
| UX Design                      |                   23.2 |                  306.2 |                   32.5 |                  493.5 |