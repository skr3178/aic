# perception_v2 — per-episode plug→port gap (all 169)

Dataset: /home/skr/aic_data/perception_v2 (169 episodes, native 1152x1024, CheatCode ground-truth demos)
"seated" = min plug→port < 6 mm.  Acceptance = valid frames (labels valid regardless of seating).

**Seated 59/169 (35%)  ·  not-seated 110 (65%)**

| ep | type | rail | target | port | frames | min_mm | final_mm | seated | retracted |
|---|---|---|---|---|---|---|---|---|---|
| ep0 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 372 | 45.8 | 46.0 | no |  |
| ep1 | sc | 1 | sc_port_1 | sc_port_base | 367 | 13.7 | 13.7 | no |  |
| ep2 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 366 | 45.8 | 45.8 | no |  |
| ep3 | sc | 1 | sc_port_1 | sc_port_base | 379 | 13.6 | 13.8 | no |  |
| ep4 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 378 | 46.2 | 46.2 | no |  |
| ep5 | sc | 1 | sc_port_1 | sc_port_base | 377 | 13.7 | 13.7 | no |  |
| ep6 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 377 | 46.3 | 46.4 | no |  |
| ep7 | sc | 0 | sc_port_0 | sc_port_base | 386 | 9.6 | 301.1 | no | yes |
| ep8 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 369 | 0.3 | 1.1 | YES |  |
| ep9 | sc | 0 | sc_port_0 | sc_port_base | 372 | 13.5 | 13.5 | no |  |
| ep10 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 355 | 0.2 | 1.0 | YES |  |
| ep11 | sc | 0 | sc_port_0 | sc_port_base | 381 | 15.4 | 17.1 | no |  |
| ep12 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 361 | 45.8 | 45.9 | no |  |
| ep13 | sc | 0 | sc_port_0 | sc_port_base | 367 | 15.8 | 15.8 | no |  |
| ep14 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 373 | 0.7 | 1.1 | YES |  |
| ep15 | sc | 0 | sc_port_0 | sc_port_base | 365 | 193.2 | 203.3 | no |  |
| ep16 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 370 | 0.4 | 1.1 | YES |  |
| ep17 | sc | 1 | sc_port_1 | sc_port_base | 376 | 44.1 | 94.8 | no | yes |
| ep18 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 370 | 0.4 | 1.1 | YES |  |
| ep19 | sc | 1 | sc_port_1 | sc_port_base | 375 | 145.8 | 163.8 | no |  |
| ep20 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 418 | 0.3 | 1.1 | YES |  |
| ep21 | sc | 1 | sc_port_1 | sc_port_base | 369 | 17.0 | 17.3 | no |  |
| ep22 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 355 | 46.2 | 46.2 | no |  |
| ep23 | sc | 0 | sc_port_0 | sc_port_base | 443 | 53.7 | 53.7 | no |  |
| ep24 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 379 | 2.1 | 2.1 | YES |  |
| ep25 | sc | 0 | sc_port_0 | sc_port_base | 393 | 13.7 | 33.5 | no |  |
| ep26 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 376 | 45.9 | 45.9 | no |  |
| ep27 | sc | 1 | sc_port_1 | sc_port_base | 382 | 16.0 | 16.0 | no |  |
| ep28 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 374 | 43.4 | 43.4 | no |  |
| ep29 | sc | 1 | sc_port_1 | sc_port_base | 385 | 13.5 | 13.5 | no |  |
| ep30 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 440 | 2.8 | 2.8 | YES |  |
| ep31 | sc | 1 | sc_port_1 | sc_port_base | 368 | 13.5 | 13.5 | no |  |
| ep32 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 440 | 0.7 | 1.1 | YES |  |
| ep33 | sc | 1 | sc_port_1 | sc_port_base | 376 | 19.7 | 22.7 | no |  |
| ep34 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 364 | 0.6 | 1.1 | YES |  |
| ep35 | sc | 0 | sc_port_0 | sc_port_base | 379 | 0.1 | 337.1 | YES | yes |
| ep36 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 376 | 44.9 | 45.0 | no |  |
| ep37 | sc | 0 | sc_port_0 | sc_port_base | 430 | 45.8 | 198.1 | no | yes |
| ep38 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 381 | 47.8 | 47.9 | no |  |
| ep39 | sc | 1 | sc_port_1 | sc_port_base | 385 | 13.8 | 13.8 | no |  |
| ep40 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 359 | 0.2 | 1.1 | YES |  |
| ep41 | sc | 0 | sc_port_0 | sc_port_base | 349 | 130.4 | 462.3 | no | yes |
| ep42 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 370 | 46.0 | 46.2 | no |  |
| ep43 | sc | 0 | sc_port_0 | sc_port_base | 376 | 30.6 | 33.8 | no |  |
| ep44 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 362 | 0.5 | 1.1 | YES |  |
| ep45 | sc | 0 | sc_port_0 | sc_port_base | 373 | 14.0 | 15.8 | no |  |
| ep46 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 367 | 1.1 | 1.1 | YES |  |
| ep47 | sc | 1 | sc_port_1 | sc_port_base | 428 | 15.0 | 15.0 | no |  |
| ep48 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 366 | 0.5 | 1.0 | YES |  |
| ep49 | sc | 1 | sc_port_1 | sc_port_base | 383 | 0.1 | 0.4 | YES |  |
| ep50 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 362 | 44.8 | 44.9 | no |  |
| ep51 | sc | 1 | sc_port_1 | sc_port_base | 365 | 20.5 | 20.6 | no |  |
| ep52 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 357 | 45.8 | 45.8 | no |  |
| ep53 | sc | 1 | sc_port_1 | sc_port_base | 354 | 0.1 | 0.3 | YES |  |
| ep54 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 367 | 0.6 | 1.0 | YES |  |
| ep55 | sc | 1 | sc_port_1 | sc_port_base | 371 | 0.3 | 0.4 | YES |  |
| ep56 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 368 | 45.8 | 45.8 | no |  |
| ep57 | sc | 0 | sc_port_0 | sc_port_base | 369 | 0.1 | 0.1 | YES |  |
| ep58 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 370 | 45.7 | 45.7 | no |  |
| ep59 | sc | 0 | sc_port_0 | sc_port_base | 408 | 13.1 | 13.1 | no |  |
| ep60 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 363 | 0.4 | 1.0 | YES |  |
| ep61 | sc | 0 | sc_port_0 | sc_port_base | 388 | 39.4 | 247.3 | no | yes |
| ep62 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 357 | 45.7 | 45.7 | no |  |
| ep63 | sc | 1 | sc_port_1 | sc_port_base | 367 | 53.4 | 60.9 | no |  |
| ep64 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 359 | 0.6 | 1.1 | YES |  |
| ep65 | sc | 0 | sc_port_0 | sc_port_base | 366 | 13.8 | 13.8 | no |  |
| ep66 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 378 | 45.8 | 45.8 | no |  |
| ep67 | sc | 0 | sc_port_0 | sc_port_base | 385 | 13.5 | 133.9 | no | yes |
| ep68 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 465 | 46.2 | 46.2 | no |  |
| ep69 | sc | 0 | sc_port_0 | sc_port_base | 394 | 13.5 | 13.5 | no |  |
| ep70 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 358 | 46.2 | 46.7 | no |  |
| ep71 | sc | 0 | sc_port_0 | sc_port_base | 367 | 17.8 | 73.0 | no | yes |
| ep72 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 362 | 45.9 | 45.9 | no |  |
| ep73 | sc | 0 | sc_port_0 | sc_port_base | 377 | 13.8 | 13.8 | no |  |
| ep74 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 362 | 0.4 | 1.0 | YES |  |
| ep75 | sc | 1 | sc_port_1 | sc_port_base | 459 | 0.1 | 0.6 | YES |  |
| ep76 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 369 | 46.3 | 46.5 | no |  |
| ep77 | sc | 1 | sc_port_1 | sc_port_base | 372 | 12.1 | 12.4 | no |  |
| ep78 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 364 | 0.4 | 1.1 | YES |  |
| ep79 | sc | 0 | sc_port_0 | sc_port_base | 391 | 0.1 | 0.1 | YES |  |
| ep80 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 367 | 0.4 | 1.0 | YES |  |
| ep81 | sc | 1 | sc_port_1 | sc_port_base | 362 | 17.6 | 17.9 | no |  |
| ep82 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 359 | 0.6 | 1.2 | YES |  |
| ep83 | sc | 0 | sc_port_0 | sc_port_base | 429 | 14.0 | 14.0 | no |  |
| ep84 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 362 | 0.6 | 1.1 | YES |  |
| ep85 | sc | 1 | sc_port_1 | sc_port_base | 390 | 8.3 | 8.3 | no |  |
| ep86 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 383 | 46.0 | 46.1 | no |  |
| ep87 | sc | 0 | sc_port_0 | sc_port_base | 389 | 13.6 | 13.6 | no |  |
| ep88 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 390 | 0.2 | 1.0 | YES |  |
| ep89 | sc | 1 | sc_port_1 | sc_port_base | 404 | 14.4 | 14.5 | no |  |
| ep90 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 368 | 0.4 | 1.0 | YES |  |
| ep91 | sc | 0 | sc_port_0 | sc_port_base | 396 | 11.5 | 543.5 | no | yes |
| ep92 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 436 | 45.8 | 45.8 | no |  |
| ep93 | sc | 1 | sc_port_1 | sc_port_base | 391 | 13.9 | 13.9 | no |  |
| ep94 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 373 | 0.4 | 1.1 | YES |  |
| ep95 | sc | 1 | sc_port_1 | sc_port_base | 401 | 14.6 | 14.6 | no |  |
| ep96 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 376 | 0.1 | 1.1 | YES |  |
| ep97 | sc | 1 | sc_port_1 | sc_port_base | 395 | 5.1 | 5.1 | YES |  |
| ep98 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 443 | 45.9 | 46.3 | no |  |
| ep99 | sc | 1 | sc_port_1 | sc_port_base | 400 | 0.2 | 0.7 | YES |  |
| ep100 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 366 | 0.4 | 1.0 | YES |  |
| ep101 | sc | 0 | sc_port_0 | sc_port_base | 377 | 15.2 | 187.1 | no | yes |
| ep102 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 369 | 0.5 | 1.0 | YES |  |
| ep103 | sc | 0 | sc_port_0 | sc_port_base | 445 | 0.2 | 230.8 | YES | yes |
| ep104 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 372 | 45.8 | 46.0 | no |  |
| ep105 | sc | 0 | sc_port_0 | sc_port_base | 375 | 0.2 | 0.2 | YES |  |
| ep106 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 378 | 0.4 | 1.1 | YES |  |
| ep107 | sc | 1 | sc_port_1 | sc_port_base | 392 | 15.4 | 15.4 | no |  |
| ep108 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 378 | 0.4 | 1.1 | YES |  |
| ep109 | sc | 0 | sc_port_0 | sc_port_base | 404 | 16.6 | 17.2 | no |  |
| ep110 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 355 | 0.5 | 1.1 | YES |  |
| ep111 | sc | 0 | sc_port_0 | sc_port_base | 441 | 0.3 | 188.8 | YES | yes |
| ep112 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 414 | 0.6 | 1.0 | YES |  |
| ep113 | sc | 0 | sc_port_0 | sc_port_base | 388 | 30.2 | 144.1 | no | yes |
| ep114 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 376 | 0.5 | 1.1 | YES |  |
| ep115 | sc | 0 | sc_port_0 | sc_port_base | 422 | 46.4 | 248.4 | no | yes |
| ep116 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 398 | 0.3 | 1.0 | YES |  |
| ep117 | sc | 1 | sc_port_1 | sc_port_base | 381 | 75.8 | 143.0 | no | yes |
| ep118 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 386 | 0.8 | 1.1 | YES |  |
| ep119 | sc | 1 | sc_port_1 | sc_port_base | 416 | 67.0 | 107.9 | no | yes |
| ep120 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 357 | 46.1 | 46.1 | no |  |
| ep121 | sc | 0 | sc_port_0 | sc_port_base | 370 | 0.1 | 133.9 | YES | yes |
| ep122 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 377 | 0.5 | 1.1 | YES |  |
| ep123 | sc | 1 | sc_port_1 | sc_port_base | 421 | 58.0 | 80.1 | no | yes |
| ep124 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 377 | 0.4 | 1.1 | YES |  |
| ep125 | sc | 1 | sc_port_1 | sc_port_base | 382 | 111.1 | 123.2 | no |  |
| ep126 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 474 | 45.9 | 46.0 | no |  |
| ep127 | sc | 1 | sc_port_1 | sc_port_base | 479 | 25.3 | 110.7 | no | yes |
| ep128 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 414 | 46.2 | 46.3 | no |  |
| ep129 | sc | 1 | sc_port_1 | sc_port_base | 427 | 0.1 | 0.1 | YES |  |
| ep130 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 355 | 45.9 | 45.9 | no |  |
| ep131 | sc | 1 | sc_port_1 | sc_port_base | 378 | 0.2 | 0.2 | YES |  |
| ep132 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 426 | 0.7 | 1.0 | YES |  |
| ep133 | sc | 0 | sc_port_0 | sc_port_base | 375 | 12.4 | 19.6 | no |  |
| ep134 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 375 | 45.8 | 45.9 | no |  |
| ep135 | sc | 0 | sc_port_0 | sc_port_base | 488 | 13.5 | 121.0 | no | yes |
| ep136 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 375 | 45.8 | 45.8 | no |  |
| ep137 | sc | 1 | sc_port_1 | sc_port_base | 410 | 5.1 | 5.1 | YES |  |
| ep138 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 401 | 44.7 | 44.7 | no |  |
| ep139 | sc | 0 | sc_port_0 | sc_port_base | 408 | 15.5 | 15.9 | no |  |
| ep140 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 356 | 45.8 | 45.8 | no |  |
| ep141 | sc | 0 | sc_port_0 | sc_port_base | 371 | 13.8 | 222.2 | no | yes |
| ep142 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 435 | 46.3 | 46.9 | no |  |
| ep143 | sc | 0 | sc_port_0 | sc_port_base | 389 | 34.3 | 34.3 | no |  |
| ep144 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 387 | 45.8 | 46.2 | no |  |
| ep145 | sc | 1 | sc_port_1 | sc_port_base | 395 | 54.3 | 56.8 | no |  |
| ep146 | sfp | 3 | nic_card_mount_3 | sfp_port_1 | 410 | 45.9 | 46.0 | no |  |
| ep147 | sc | 1 | sc_port_1 | sc_port_base | 435 | 163.9 | 416.7 | no | yes |
| ep148 | sfp | 2 | nic_card_mount_2 | sfp_port_0 | 404 | 46.3 | 47.0 | no |  |
| ep149 | sc | 1 | sc_port_1 | sc_port_base | 409 | 13.6 | 13.6 | no |  |
| ep150 | sfp | 2 | nic_card_mount_2 | sfp_port_1 | 385 | 45.8 | 45.8 | no |  |
| ep151 | sc | 1 | sc_port_1 | sc_port_base | 428 | 19.3 | 19.3 | no |  |
| ep152 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 390 | 0.5 | 1.1 | YES |  |
| ep153 | sc | 1 | sc_port_1 | sc_port_base | 408 | 13.8 | 13.9 | no |  |
| ep154 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 394 | 46.4 | 47.0 | no |  |
| ep155 | sc | 0 | sc_port_0 | sc_port_base | 375 | 0.1 | 0.2 | YES |  |
| ep156 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 376 | 44.6 | 44.6 | no |  |
| ep157 | sc | 0 | sc_port_0 | sc_port_base | 413 | 17.3 | 17.4 | no |  |
| ep158 | sfp | 3 | nic_card_mount_3 | sfp_port_0 | 393 | 45.0 | 45.1 | no |  |
| ep159 | sc | 1 | sc_port_1 | sc_port_base | 418 | 0.2 | 0.2 | YES |  |
| ep160 | sfp | 0 | nic_card_mount_0 | sfp_port_1 | 380 | 0.6 | 1.1 | YES |  |
| ep161 | sc | 1 | sc_port_1 | sc_port_base | 401 | 14.2 | 14.2 | no |  |
| ep162 | sfp | 1 | nic_card_mount_1 | sfp_port_1 | 31 | 128.0 | 128.0 | no |  |
| ep164 | sfp | 4 | nic_card_mount_4 | sfp_port_1 | 301 | 0.4 | 1.1 | YES |  |
| ep170 | sfp | 0 | nic_card_mount_0 | sfp_port_0 | 397 | 45.9 | 46.5 | no |  |
| ep171 | sc | 0 | sc_port_0 | sc_port_base | 404 | 46.6 | 422.3 | no | yes |
| ep172 | sfp | 1 | nic_card_mount_1 | sfp_port_0 | 399 | 43.4 | 43.4 | no |  |
| ep173 | sc | 0 | sc_port_0 | sc_port_base | 436 | 34.6 | 39.5 | no |  |
| ep174 | sfp | 4 | nic_card_mount_4 | sfp_port_0 | 132 | 96.7 | 96.7 | no |  |
