# Effect Report (newlayout, mean only)

Source: *_newlayout/*/summaries/sentiment_significance.csv
Values are mean across runs. Metrics: CramersV / TV / JS.
Bold = max/min within each table (per metric).

## View 1: By model (rows = dimensions, grouped by dataset)

### Model: llama-3b_newlayout

#### Dataset: fiqasa

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | **0.401070** | **0.315857** | 0.111642 |
| ST-NF | ST | 0.165523 | 0.068627 | 0.003559 |
| decision | F | 0.290337 | 0.168372 | 0.027484 |
| decision | T | 0.229750 | 0.101876 | 0.015292 |
| energy | E | 0.399981 | 0.291134 | **0.123816** |
| energy | I | **0.134901** | 0.062234 | 0.003859 |
| execution | J | 0.357880 | 0.230179 | 0.076822 |
| execution | P | 0.219045 | 0.092072 | 0.010883 |
| information | N | 0.154781 | **0.054987** | **0.003387** |
| information | S | 0.354225 | 0.242540 | 0.079660 |

#### Dataset: imdb

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | **0.166468** | **0.055350** | 0.003101 |
| ST-NF | ST | 0.161049 | 0.052100 | 0.004121 |
| decision | F | 0.103422 | 0.022400 | 0.000700 |
| decision | T | 0.140760 | 0.039600 | 0.003208 |
| energy | E | 0.075052 | 0.012100 | **0.000126** |
| energy | I | 0.075800 | 0.012100 | 0.000191 |
| execution | J | **0.050364** | **0.007150** | 0.000165 |
| execution | P | 0.073620 | 0.012150 | 0.000927 |
| information | N | 0.116961 | 0.023550 | **0.007011** |
| information | S | 0.117217 | 0.028950 | 0.001148 |

#### Dataset: imdb_sklearn

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | **0.179049** | **0.064750** | 0.005217 |
| ST-NF | ST | 0.159301 | 0.050850 | **0.005275** |
| decision | F | 0.075578 | 0.013900 | 0.000143 |
| decision | T | 0.151143 | 0.045750 | 0.001537 |
| energy | E | 0.068604 | 0.011050 | 0.000138 |
| energy | I | 0.043966 | 0.006150 | **0.000078** |
| execution | J | **0.032341** | **0.003850** | 0.000733 |
| execution | P | 0.078328 | 0.016450 | 0.003061 |
| information | N | 0.074098 | 0.014900 | 0.000210 |
| information | S | 0.127331 | 0.033600 | 0.000824 |

#### Dataset: mental

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | 0.057975 | 0.014955 | 0.000357 |
| ST-NF | ST | 0.543085 | 0.296263 | 0.079712 |
| decision | F | 0.204815 | 0.045089 | 0.004197 |
| decision | T | **0.032045** | **0.004310** | **0.000028** |
| energy | E | 0.194299 | 0.038996 | 0.002032 |
| energy | I | 0.183789 | 0.034564 | 0.002004 |
| execution | J | 0.157760 | 0.045828 | 0.002743 |
| execution | P | **0.597144** | **0.357459** | **0.110073** |
| information | N | 0.205354 | 0.043183 | 0.003197 |
| information | S | 0.098792 | 0.016651 | 0.000392 |

#### Dataset: news

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | **0.381308** | **0.291258** | **0.065698** |
| ST-NF | ST | 0.303280 | 0.183975 | 0.029612 |
| decision | F | 0.169614 | 0.066885 | 0.004848 |
| decision | T | 0.217222 | 0.096890 | 0.007280 |
| energy | E | 0.249035 | 0.121574 | 0.014364 |
| energy | I | 0.267519 | 0.147263 | 0.018174 |
| execution | J | 0.116694 | 0.036124 | 0.001348 |
| execution | P | **0.063775** | **0.023049** | **0.000634** |
| information | N | 0.268624 | 0.151664 | 0.017776 |
| information | S | 0.232343 | 0.093915 | 0.008385 |

#### Dataset: sst2

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| ST-NF | NF | **0.269867** | **0.142550** | 0.014959 |
| ST-NF | ST | 0.262963 | 0.137350 | 0.016796 |
| decision | F | 0.197873 | 0.076400 | 0.004720 |
| decision | T | 0.236264 | 0.110650 | 0.015320 |
| energy | E | 0.184940 | 0.063650 | 0.007517 |
| energy | I | **0.144100** | **0.041350** | **0.002281** |
| execution | J | 0.168062 | 0.055800 | 0.007738 |
| execution | P | 0.204950 | 0.071100 | 0.011733 |
| information | N | 0.206441 | 0.075650 | 0.012756 |
| information | S | 0.242348 | 0.117800 | **0.023130** |

### Model: qwen-3b_newlayout

#### Dataset: fiqasa

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.191283 | 0.070759 | 0.007163 |
| decision | T | **0.203698** | **0.080989** | **0.008965** |
| energy | E | 0.188098 | 0.069906 | 0.005569 |
| energy | I | 0.160017 | 0.050298 | 0.003415 |
| execution | J | 0.184352 | 0.062234 | 0.007455 |
| execution | P | 0.148557 | 0.043478 | 0.002852 |
| information | N | **0.066743** | **0.011083** | **0.000094** |
| information | S | 0.099762 | 0.019608 | 0.000671 |

#### Dataset: imdb

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | **0.089443** | **0.016000** | **0.000185** |
| decision | T | 0.079373 | 0.012600 | 0.000115 |
| energy | E | 0.078102 | 0.012200 | 0.000108 |
| energy | I | 0.083066 | 0.013800 | 0.000138 |
| execution | J | **0.011785** | **0.001000** | **0.000001** |
| execution | P | 0.019050 | 0.001500 | 0.000002 |
| information | N | 0.063116 | 0.008700 | 0.000055 |
| information | S | 0.062610 | 0.008400 | 0.000051 |

#### Dataset: imdb_sklearn

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.068406 | 0.010100 | 0.000074 |
| decision | T | 0.074845 | 0.011400 | 0.000314 |
| energy | E | **0.110087** | **0.025200** | **0.000462** |
| energy | I | **0.003307** | **0.000700** | **0.000000** |
| execution | J | 0.073302 | 0.014100 | 0.000145 |
| execution | P | 0.068104 | 0.013100 | 0.000125 |
| information | N | 0.086048 | 0.017700 | 0.000239 |
| information | S | 0.052926 | 0.009700 | 0.000118 |

#### Dataset: mental

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.238577 | 0.056919 | 0.002342 |
| decision | T | 0.191646 | 0.036854 | 0.000983 |
| energy | E | 0.199978 | 0.050556 | 0.001847 |
| energy | I | 0.234186 | 0.063061 | 0.002877 |
| execution | J | 0.229939 | 0.061675 | 0.002751 |
| execution | P | 0.210492 | 0.054714 | 0.002164 |
| information | N | **0.315746** | **0.104331** | **0.007929** |
| information | S | **0.077141** | **0.016538** | **0.000197** |

#### Dataset: news

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.191277 | 0.073338 | 0.005154 |
| decision | T | 0.147625 | 0.041824 | 0.001773 |
| energy | E | 0.056152 | **0.010058** | 0.000116 |
| energy | I | **0.224886** | **0.100997** | **0.008040** |
| execution | J | **0.052063** | 0.011064 | **0.000091** |
| execution | P | 0.219036 | 0.095298 | 0.007007 |
| information | N | 0.145178 | 0.036124 | 0.002002 |
| information | S | 0.179099 | 0.062191 | 0.003429 |

#### Dataset: sst2

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | **0.168693** | **0.056900** | **0.002734** |
| decision | T | 0.163912 | 0.053600 | 0.002620 |
| energy | E | 0.150000 | 0.045000 | 0.001660 |
| energy | I | 0.153158 | 0.046900 | 0.001940 |
| execution | J | 0.079286 | 0.012800 | 0.000304 |
| execution | P | 0.084715 | 0.014000 | 0.000269 |
| information | N | **0.004352** | **0.000600** | **0.000002** |
| information | S | 0.034650 | 0.005600 | 0.000047 |

### Model: qwen-7b_newlayout

#### Dataset: fiqasa

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.143141 | 0.042626 | 0.001780 |
| decision | T | **0.160367** | **0.052856** | **0.002422** |
| energy | E | 0.138685 | 0.038363 | 0.001341 |
| energy | I | 0.132199 | 0.036658 | 0.001232 |
| execution | J | 0.118602 | 0.028133 | 0.000690 |
| execution | P | 0.123876 | 0.030691 | 0.000787 |
| information | N | **0.114867** | 0.030691 | 0.000815 |
| information | S | 0.116791 | **0.027280** | **0.000618** |

#### Dataset: imdb

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.069150 | 0.009600 | 0.000119 |
| decision | T | **0.082617** | **0.013400** | **0.000141** |
| energy | E | 0.052989 | 0.005600 | 0.000037 |
| energy | I | 0.059417 | 0.006800 | 0.000038 |
| execution | J | **0.021361** | 0.001500 | 0.000024 |
| execution | P | 0.023498 | **0.001400** | **0.000010** |
| information | N | 0.031724 | 0.002600 | 0.000063 |
| information | S | 0.044314 | 0.004600 | 0.000082 |

#### Dataset: imdb_sklearn

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.072471 | 0.010500 | 0.000390 |
| decision | T | **0.083913** | **0.013900** | **0.000149** |
| energy | E | 0.075081 | 0.013000 | 0.000581 |
| energy | I | 0.035469 | **0.003800** | 0.000444 |
| execution | J | 0.047944 | 0.006700 | 0.000437 |
| execution | P | 0.064344 | 0.010500 | 0.000693 |
| information | N | 0.077165 | 0.013800 | **0.001307** |
| information | S | **0.033879** | 0.004100 | 0.000235 |

#### Dataset: mental

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.188668 | 0.035753 | 0.000993 |
| decision | T | 0.184917 | 0.034257 | 0.000881 |
| energy | E | 0.171220 | 0.041542 | 0.001346 |
| energy | I | 0.226267 | 0.060815 | 0.002922 |
| execution | J | 0.217849 | 0.057574 | 0.002613 |
| execution | P | 0.165324 | 0.040066 | 0.001251 |
| information | N | **0.278247** | **0.082457** | **0.005454** |
| information | S | **0.097494** | **0.022371** | **0.000386** |

#### Dataset: news

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | **0.146180** | **0.049074** | **0.001897** |
| decision | T | 0.141162 | 0.032856 | 0.000912 |
| energy | E | 0.095635 | 0.019361 | 0.000277 |
| energy | I | 0.123852 | 0.029168 | 0.000770 |
| execution | J | 0.113137 | 0.022882 | 0.000457 |
| execution | P | 0.132460 | 0.035538 | 0.000950 |
| information | N | **0.012919** | **0.002682** | **0.000006** |
| information | S | 0.076614 | 0.014919 | 0.000193 |

#### Dataset: sst2

| Dimension | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| decision | F | 0.128460 | 0.032500 | 0.001434 |
| decision | T | **0.138151** | **0.038300** | **0.001843** |
| energy | E | 0.087980 | 0.014200 | 0.000160 |
| energy | I | 0.087980 | 0.014200 | 0.000159 |
| execution | J | 0.063419 | 0.006900 | 0.000196 |
| execution | P | **0.040907** | **0.004650** | **0.000080** |
| information | N | 0.077217 | 0.012600 | 0.000134 |
| information | S | 0.086308 | 0.014600 | 0.000359 |

## View 2: By dimension (rows = models, grouped by dataset)

### Dimension: ST-NF

#### Dataset: fiqasa

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.401070** | **0.315857** | **0.111642** |
| llama-3b_newlayout | ST | **0.165523** | **0.068627** | **0.003559** |

#### Dataset: imdb

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.166468** | **0.055350** | **0.003101** |
| llama-3b_newlayout | ST | **0.161049** | **0.052100** | **0.004121** |

#### Dataset: imdb_sklearn

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.179049** | **0.064750** | **0.005217** |
| llama-3b_newlayout | ST | **0.159301** | **0.050850** | **0.005275** |

#### Dataset: mental

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.057975** | **0.014955** | **0.000357** |
| llama-3b_newlayout | ST | **0.543085** | **0.296263** | **0.079712** |

#### Dataset: news

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.381308** | **0.291258** | **0.065698** |
| llama-3b_newlayout | ST | **0.303280** | **0.183975** | **0.029612** |

#### Dataset: sst2

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | NF | **0.269867** | **0.142550** | **0.014959** |
| llama-3b_newlayout | ST | **0.262963** | **0.137350** | **0.016796** |

### Dimension: decision

#### Dataset: fiqasa

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | **0.290337** | **0.168372** | **0.027484** |
| llama-3b_newlayout | T | 0.229750 | 0.101876 | 0.015292 |
| qwen-3b_newlayout | F | 0.191283 | 0.070759 | 0.007163 |
| qwen-3b_newlayout | T | 0.203698 | 0.080989 | 0.008965 |
| qwen-7b_newlayout | F | **0.143141** | **0.042626** | **0.001780** |
| qwen-7b_newlayout | T | 0.160367 | 0.052856 | 0.002422 |

#### Dataset: imdb

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | 0.103422 | 0.022400 | 0.000700 |
| llama-3b_newlayout | T | **0.140760** | **0.039600** | **0.003208** |
| qwen-3b_newlayout | F | 0.089443 | 0.016000 | 0.000185 |
| qwen-3b_newlayout | T | 0.079373 | 0.012600 | **0.000115** |
| qwen-7b_newlayout | F | **0.069150** | **0.009600** | 0.000119 |
| qwen-7b_newlayout | T | 0.082617 | 0.013400 | 0.000141 |

#### Dataset: imdb_sklearn

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | 0.075578 | 0.013900 | 0.000143 |
| llama-3b_newlayout | T | **0.151143** | **0.045750** | **0.001537** |
| qwen-3b_newlayout | F | **0.068406** | **0.010100** | **0.000074** |
| qwen-3b_newlayout | T | 0.074845 | 0.011400 | 0.000314 |
| qwen-7b_newlayout | F | 0.072471 | 0.010500 | 0.000390 |
| qwen-7b_newlayout | T | 0.083913 | 0.013900 | 0.000149 |

#### Dataset: mental

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | 0.204815 | 0.045089 | **0.004197** |
| llama-3b_newlayout | T | **0.032045** | **0.004310** | **0.000028** |
| qwen-3b_newlayout | F | **0.238577** | **0.056919** | 0.002342 |
| qwen-3b_newlayout | T | 0.191646 | 0.036854 | 0.000983 |
| qwen-7b_newlayout | F | 0.188668 | 0.035753 | 0.000993 |
| qwen-7b_newlayout | T | 0.184917 | 0.034257 | 0.000881 |

#### Dataset: news

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | 0.169614 | 0.066885 | 0.004848 |
| llama-3b_newlayout | T | **0.217222** | **0.096890** | **0.007280** |
| qwen-3b_newlayout | F | 0.191277 | 0.073338 | 0.005154 |
| qwen-3b_newlayout | T | 0.147625 | 0.041824 | 0.001773 |
| qwen-7b_newlayout | F | 0.146180 | 0.049074 | 0.001897 |
| qwen-7b_newlayout | T | **0.141162** | **0.032856** | **0.000912** |

#### Dataset: sst2

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | F | 0.197873 | 0.076400 | 0.004720 |
| llama-3b_newlayout | T | **0.236264** | **0.110650** | **0.015320** |
| qwen-3b_newlayout | F | 0.168693 | 0.056900 | 0.002734 |
| qwen-3b_newlayout | T | 0.163912 | 0.053600 | 0.002620 |
| qwen-7b_newlayout | F | **0.128460** | **0.032500** | **0.001434** |
| qwen-7b_newlayout | T | 0.138151 | 0.038300 | 0.001843 |

### Dimension: energy

#### Dataset: fiqasa

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | **0.399981** | **0.291134** | **0.123816** |
| llama-3b_newlayout | I | 0.134901 | 0.062234 | 0.003859 |
| qwen-3b_newlayout | E | 0.188098 | 0.069906 | 0.005569 |
| qwen-3b_newlayout | I | 0.160017 | 0.050298 | 0.003415 |
| qwen-7b_newlayout | E | 0.138685 | 0.038363 | 0.001341 |
| qwen-7b_newlayout | I | **0.132199** | **0.036658** | **0.001232** |

#### Dataset: imdb

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | 0.075052 | 0.012100 | 0.000126 |
| llama-3b_newlayout | I | 0.075800 | 0.012100 | **0.000191** |
| qwen-3b_newlayout | E | 0.078102 | 0.012200 | 0.000108 |
| qwen-3b_newlayout | I | **0.083066** | **0.013800** | 0.000138 |
| qwen-7b_newlayout | E | **0.052989** | **0.005600** | **0.000037** |
| qwen-7b_newlayout | I | 0.059417 | 0.006800 | 0.000038 |

#### Dataset: imdb_sklearn

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | 0.068604 | 0.011050 | 0.000138 |
| llama-3b_newlayout | I | 0.043966 | 0.006150 | 0.000078 |
| qwen-3b_newlayout | E | **0.110087** | **0.025200** | 0.000462 |
| qwen-3b_newlayout | I | **0.003307** | **0.000700** | **0.000000** |
| qwen-7b_newlayout | E | 0.075081 | 0.013000 | **0.000581** |
| qwen-7b_newlayout | I | 0.035469 | 0.003800 | 0.000444 |

#### Dataset: mental

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | 0.194299 | 0.038996 | 0.002032 |
| llama-3b_newlayout | I | 0.183789 | **0.034564** | 0.002004 |
| qwen-3b_newlayout | E | 0.199978 | 0.050556 | 0.001847 |
| qwen-3b_newlayout | I | **0.234186** | **0.063061** | 0.002877 |
| qwen-7b_newlayout | E | **0.171220** | 0.041542 | **0.001346** |
| qwen-7b_newlayout | I | 0.226267 | 0.060815 | **0.002922** |

#### Dataset: news

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | 0.249035 | 0.121574 | 0.014364 |
| llama-3b_newlayout | I | **0.267519** | **0.147263** | **0.018174** |
| qwen-3b_newlayout | E | **0.056152** | **0.010058** | **0.000116** |
| qwen-3b_newlayout | I | 0.224886 | 0.100997 | 0.008040 |
| qwen-7b_newlayout | E | 0.095635 | 0.019361 | 0.000277 |
| qwen-7b_newlayout | I | 0.123852 | 0.029168 | 0.000770 |

#### Dataset: sst2

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | E | **0.184940** | **0.063650** | **0.007517** |
| llama-3b_newlayout | I | 0.144100 | 0.041350 | 0.002281 |
| qwen-3b_newlayout | E | 0.150000 | 0.045000 | 0.001660 |
| qwen-3b_newlayout | I | 0.153158 | 0.046900 | 0.001940 |
| qwen-7b_newlayout | E | **0.087980** | **0.014200** | 0.000160 |
| qwen-7b_newlayout | I | **0.087980** | 0.014200 | **0.000159** |

### Dimension: execution

#### Dataset: fiqasa

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | **0.357880** | **0.230179** | **0.076822** |
| llama-3b_newlayout | P | 0.219045 | 0.092072 | 0.010883 |
| qwen-3b_newlayout | J | 0.184352 | 0.062234 | 0.007455 |
| qwen-3b_newlayout | P | 0.148557 | 0.043478 | 0.002852 |
| qwen-7b_newlayout | J | **0.118602** | **0.028133** | **0.000690** |
| qwen-7b_newlayout | P | 0.123876 | 0.030691 | 0.000787 |

#### Dataset: imdb

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | 0.050364 | 0.007150 | 0.000165 |
| llama-3b_newlayout | P | **0.073620** | **0.012150** | **0.000927** |
| qwen-3b_newlayout | J | **0.011785** | **0.001000** | **0.000001** |
| qwen-3b_newlayout | P | 0.019050 | 0.001500 | 0.000002 |
| qwen-7b_newlayout | J | 0.021361 | 0.001500 | 0.000024 |
| qwen-7b_newlayout | P | 0.023498 | 0.001400 | 0.000010 |

#### Dataset: imdb_sklearn

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | **0.032341** | **0.003850** | 0.000733 |
| llama-3b_newlayout | P | **0.078328** | **0.016450** | **0.003061** |
| qwen-3b_newlayout | J | 0.073302 | 0.014100 | 0.000145 |
| qwen-3b_newlayout | P | 0.068104 | 0.013100 | **0.000125** |
| qwen-7b_newlayout | J | 0.047944 | 0.006700 | 0.000437 |
| qwen-7b_newlayout | P | 0.064344 | 0.010500 | 0.000693 |

#### Dataset: mental

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | **0.157760** | 0.045828 | 0.002743 |
| llama-3b_newlayout | P | **0.597144** | **0.357459** | **0.110073** |
| qwen-3b_newlayout | J | 0.229939 | 0.061675 | 0.002751 |
| qwen-3b_newlayout | P | 0.210492 | 0.054714 | 0.002164 |
| qwen-7b_newlayout | J | 0.217849 | 0.057574 | 0.002613 |
| qwen-7b_newlayout | P | 0.165324 | **0.040066** | **0.001251** |

#### Dataset: news

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | 0.116694 | 0.036124 | 0.001348 |
| llama-3b_newlayout | P | 0.063775 | 0.023049 | 0.000634 |
| qwen-3b_newlayout | J | **0.052063** | **0.011064** | **0.000091** |
| qwen-3b_newlayout | P | **0.219036** | **0.095298** | **0.007007** |
| qwen-7b_newlayout | J | 0.113137 | 0.022882 | 0.000457 |
| qwen-7b_newlayout | P | 0.132460 | 0.035538 | 0.000950 |

#### Dataset: sst2

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | J | 0.168062 | 0.055800 | 0.007738 |
| llama-3b_newlayout | P | **0.204950** | **0.071100** | **0.011733** |
| qwen-3b_newlayout | J | 0.079286 | 0.012800 | 0.000304 |
| qwen-3b_newlayout | P | 0.084715 | 0.014000 | 0.000269 |
| qwen-7b_newlayout | J | 0.063419 | 0.006900 | 0.000196 |
| qwen-7b_newlayout | P | **0.040907** | **0.004650** | **0.000080** |

### Dimension: information

#### Dataset: fiqasa

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | 0.154781 | 0.054987 | 0.003387 |
| llama-3b_newlayout | S | **0.354225** | **0.242540** | **0.079660** |
| qwen-3b_newlayout | N | **0.066743** | **0.011083** | **0.000094** |
| qwen-3b_newlayout | S | 0.099762 | 0.019608 | 0.000671 |
| qwen-7b_newlayout | N | 0.114867 | 0.030691 | 0.000815 |
| qwen-7b_newlayout | S | 0.116791 | 0.027280 | 0.000618 |

#### Dataset: imdb

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | 0.116961 | 0.023550 | **0.007011** |
| llama-3b_newlayout | S | **0.117217** | **0.028950** | 0.001148 |
| qwen-3b_newlayout | N | 0.063116 | 0.008700 | 0.000055 |
| qwen-3b_newlayout | S | 0.062610 | 0.008400 | **0.000051** |
| qwen-7b_newlayout | N | **0.031724** | **0.002600** | 0.000063 |
| qwen-7b_newlayout | S | 0.044314 | 0.004600 | 0.000082 |

#### Dataset: imdb_sklearn

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | 0.074098 | 0.014900 | 0.000210 |
| llama-3b_newlayout | S | **0.127331** | **0.033600** | 0.000824 |
| qwen-3b_newlayout | N | 0.086048 | 0.017700 | 0.000239 |
| qwen-3b_newlayout | S | 0.052926 | 0.009700 | **0.000118** |
| qwen-7b_newlayout | N | 0.077165 | 0.013800 | **0.001307** |
| qwen-7b_newlayout | S | **0.033879** | **0.004100** | 0.000235 |

#### Dataset: mental

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | 0.205354 | 0.043183 | 0.003197 |
| llama-3b_newlayout | S | 0.098792 | 0.016651 | 0.000392 |
| qwen-3b_newlayout | N | **0.315746** | **0.104331** | **0.007929** |
| qwen-3b_newlayout | S | **0.077141** | **0.016538** | **0.000197** |
| qwen-7b_newlayout | N | 0.278247 | 0.082457 | 0.005454 |
| qwen-7b_newlayout | S | 0.097494 | 0.022371 | 0.000386 |

#### Dataset: news

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | **0.268624** | **0.151664** | **0.017776** |
| llama-3b_newlayout | S | 0.232343 | 0.093915 | 0.008385 |
| qwen-3b_newlayout | N | 0.145178 | 0.036124 | 0.002002 |
| qwen-3b_newlayout | S | 0.179099 | 0.062191 | 0.003429 |
| qwen-7b_newlayout | N | **0.012919** | **0.002682** | **0.000006** |
| qwen-7b_newlayout | S | 0.076614 | 0.014919 | 0.000193 |

#### Dataset: sst2

| Model | Model code | CramersV | TV | JS |
|---|---|---:|---:|---:|
| llama-3b_newlayout | N | 0.206441 | 0.075650 | 0.012756 |
| llama-3b_newlayout | S | **0.242348** | **0.117800** | **0.023130** |
| qwen-3b_newlayout | N | **0.004352** | **0.000600** | **0.000002** |
| qwen-3b_newlayout | S | 0.034650 | 0.005600 | 0.000047 |
| qwen-7b_newlayout | N | 0.077217 | 0.012600 | 0.000134 |
| qwen-7b_newlayout | S | 0.086308 | 0.014600 | 0.000359 |
