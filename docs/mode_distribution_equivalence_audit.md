# Mode/Ticket Distribution Equivalence Audit

This audit reuses existing posterior artifacts and treats each row as a
distributional support-overlap test against the matching IMP ticket.
A posterior-mode rescue would require posterior-to-IMP overlap to exceed
matched chain-start, dense, or rewind magnitude controls, not merely
uniform random masks.

Summary:

- Posterior beats random in 58/59 grouped comparisons.
- Posterior beats chain-start by more than 0.005 Jaccard in 0/59 grouped comparisons.
- Posterior is practically tied to chain-start in 43/59 grouped comparisons.
- Posterior is below chain-start by more than 0.005 Jaccard in 16/59 grouped comparisons.
- Rewind magnitude beats posterior by more than 0.005 Jaccard in 55/57 grouped comparisons.

## Critical Posterior-vs-Chain Comparisons

| Family | Config | Scope | n | Posterior | Chain | Delta | Wins | KS p | W-dist | Post-chain | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | block | 5 | 0.3294 | 0.3501 | -0.0206 | 0.0000 | 0.0800 | 0.0206 | 0.5088 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-03 | global | 5 | 0.1447 | 0.1469 | -0.0021 | 0.0000 | 0.3200 | 0.0021 | 0.8826 | practically tied to control |
| CIFAR HeadLap | scale=1.0e-03 | head | 5 | 0.6983 | 0.7068 | -0.0085 | 0.4000 | 0.8200 | 0.0125 | 0.7773 | mixed |
| CIFAR Hess16SubHMC | step=3.0e-04 | global | 5 | 0.1468 | 0.1468 | -1.64e-05 | 0.4000 | 1.0000 | 3.75e-05 | 0.9994 | practically tied to control |
| CIFAR Hess32SubHMC | step=3.0e-04 | global | 5 | 0.1461 | 0.1461 | 3.55e-05 | 0.8000 | 1.0000 | 3.75e-05 | 0.9993 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-04 | global | 5 | 0.1456 | 0.1456 | -2.16e-05 | 0.6000 | 1.0000 | 0.0002 | 0.9334 | practically tied to control |
| CIFAR LowRank128Lap movement | scale=1.0e-02 | global | 5 | 0.1351 | 0.1453 | -0.0102 | 0.0000 | 0.0000 | 0.0102 | 0.7358 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-02 | global | 5 | 0.1358 | 0.1457 | -0.0098 | 0.0000 | 0.0000 | 0.0098 | 0.7402 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-02 | global | 5 | 0.1339 | 0.1433 | -0.0095 | 0.0000 | 0.0000 | 0.0095 | 0.7397 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-02 | global | 5 | 0.1351 | 0.1456 | -0.0105 | 0.0000 | 0.0000 | 0.0105 | 0.7359 | control closer to ticket |
| CIFAR RandSubHMC | step=3.0e-03 | global | 5 | 0.1440 | 0.1440 | -2.23e-05 | 0.2000 | 0.8200 | 6.82e-05 | 0.9766 | practically tied to control |
| CIFAR SGHMC movement | lr=1.0e-07 | global | 5 | 0.1419 | 0.1457 | -0.0037 | 0.0000 | 0.3200 | 0.0037 | 0.6796 | practically tied to control |
| CIFAR SGLD movement | lr=1.0e-06 | global | 5 | 0.1425 | 0.1441 | -0.0017 | 0.0000 | 0.8200 | 0.0017 | 0.7362 | practically tied to control |
| CIFAR SGLD-3chain | 30ep rewind r5 p0.30 | global | 5 | 0.1368 | 0.1368 | -4.88e-06 | 0.4000 | 1.0000 | 3.37e-05 | 0.9969 | practically tied to control |
| CIFAR SWAG | 30ep rewind r5 p0.30 | global | 5 | 0.1361 | 0.1361 | -5.38e-05 | 0.6000 | 0.8200 | 0.0003 | 0.9265 | practically tied to control |
| CIFAR SWAG20 movement | scale=1.6e+01 | global | 5 | 0.1454 | 0.1455 | -7.02e-05 | 0.2000 | 1.0000 | 0.0001 | 0.9528 | practically tied to control |
| CIFAR TrajSubHMC | step=1.0e-03 | global | 5 | 0.2290 | 0.2292 | -0.0002 | 0.4000 | 1.0000 | 0.0002 | 0.9915 | practically tied to control |
| CIFAR cSGLD movement | lr=1.0e-06 | global | 5 | 0.1422 | 0.1454 | -0.0033 | 0.0000 | 0.0800 | 0.0033 | 0.7046 | practically tied to control |
| Fashion Gate1 | r5 p0.30 | global | 5 | 0.2114 | 0.2122 | -0.0008 | 0.0000 | 0.3200 | 0.0008 | 0.9212 | practically tied to control |
| MNIST Gate1 | r5 p0.30 | global | 5 | 0.2373 | 0.2385 | -0.0011 | 0.0000 | 0.3200 | 0.0011 | 0.9298 | practically tied to control |

## All Grouped Comparisons

| Family | Config | Scope | Comparison | n | Posterior | Baseline | Delta | 95% CI | Wins | KS p | W-dist | MMD | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | block | posterior-random | 5 | 0.3294 | 0.2710 | 0.0584 | [0.0532, 0.0636] | 1.0000 | 0.0000 | 0.0584 | 1.0355 | posterior separates from random |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | block | posterior-chain | 5 | 0.3294 | 0.3501 | -0.0206 | [-0.0253, -0.0159] | 0.0000 | 0.0800 | 0.0206 | 0.9056 | control closer to ticket |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | block | posterior-rewind | 5 | 0.3294 | 0.3637 | -0.0343 | [-0.0405, -0.0280] | 0.0000 | 0.0000 | 0.0343 | 1.0408 | control closer to ticket |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | global | posterior-random | 5 | 0.1319 | 0.0918 | 0.0401 | [0.0381, 0.0422] | 1.0000 | 0.0000 | 0.0401 | 0.8894 | posterior separates from random |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | global | posterior-chain | 5 | 0.1319 | 0.1309 | 0.0010 | [0.0008, 0.0012] | 1.0000 | 0.8200 | 0.0010 | 0.0453 | practically tied to control |
| CIFAR BlockLap | joint-group, scale=1.0e-04 | global | posterior-rewind | 5 | 0.1319 | 0.1509 | -0.0190 | [-0.0219, -0.0161] | 0.0000 | 0.0000 | 0.0190 | 1.2940 | control closer to ticket |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | block | posterior-random | 5 | 0.1959 | 0.1938 | 0.0021 | [-0.0025, 0.0068] | 0.8000 | 1.0000 | 0.0050 | 0.0194 | posterior separates from random |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | block | posterior-chain | 5 | 0.1959 | 0.2034 | -0.0075 | [-0.0159, 0.0009] | 0.0000 | 0.8200 | 0.0075 | 0.0763 | control closer to ticket |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | block | posterior-rewind | 5 | 0.1959 | 0.2423 | -0.0464 | [-0.0543, -0.0385] | 0.0000 | 0.0800 | 0.0464 | 0.8524 | control closer to ticket |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | global | posterior-random | 5 | 0.1329 | 0.0917 | 0.0411 | [0.0386, 0.0436] | 1.0000 | 0.0000 | 0.0411 | 0.8667 | posterior separates from random |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | global | posterior-chain | 5 | 0.1329 | 0.1315 | 0.0013 | [0.0010, 0.0016] | 1.0000 | 0.8200 | 0.0013 | 0.0503 | practically tied to control |
| CIFAR BlockLap | layer1.0.conv1, scale=1.0e-03 | global | posterior-rewind | 5 | 0.1329 | 0.1504 | -0.0176 | [-0.0197, -0.0155] | 0.0000 | 0.0000 | 0.0176 | 1.1867 | control closer to ticket |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | block | posterior-random | 5 | 0.2402 | 0.2381 | 0.0021 | [-0.0008, 0.0050] | 0.8000 | 0.3200 | 0.0034 | 0.1317 | posterior separates from random |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | block | posterior-chain | 5 | 0.2402 | 0.2411 | -0.0010 | [-0.0108, 0.0089] | 0.4000 | 0.8200 | 0.0069 | 0.1356 | practically tied to control |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | block | posterior-rewind | 5 | 0.2402 | 0.3050 | -0.0648 | [-0.0800, -0.0497] | 0.0000 | 0.0000 | 0.0648 | 1.2371 | control closer to ticket |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | global | posterior-random | 5 | 0.1333 | 0.0918 | 0.0414 | [0.0399, 0.0430] | 1.0000 | 0.0000 | 0.0414 | 0.8628 | posterior separates from random |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | global | posterior-chain | 5 | 0.1333 | 0.1324 | 0.0008 | [0.0006, 0.0010] | 1.0000 | 0.8200 | 0.0008 | 0.0394 | practically tied to control |
| CIFAR BlockLap | layer3.0.shortcut.0, scale=1.0e-03 | global | posterior-rewind | 5 | 0.1333 | 0.1513 | -0.0181 | [-0.0222, -0.0140] | 0.0000 | 0.0000 | 0.0181 | 1.1836 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1278 | 0.0918 | 0.0360 | [0.0340, 0.0380] | 1.0000 | 0.0000 | 0.0360 | 0.8897 | posterior separates from random |
| CIFAR DiagLap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1278 | 0.1469 | -0.0191 | [-0.0209, -0.0173] | 0.0000 | 0.0000 | 0.0191 | 1.0363 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1278 | 0.1469 | -0.0191 | [-0.0209, -0.0173] | 0.0000 | 0.0000 | 0.0191 | 1.0363 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1278 | 0.1787 | -0.0509 | [-0.0527, -0.0490] | 0.0000 | 0.0000 | 0.0509 | 0.9183 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-03 | global | posterior-random | 5 | 0.1447 | 0.0918 | 0.0530 | [0.0511, 0.0548] | 1.0000 | 0.0000 | 0.0530 | 0.8438 | posterior separates from random |
| CIFAR DiagLap movement | scale=1.0e-03 | global | posterior-chain | 5 | 0.1447 | 0.1469 | -0.0021 | [-0.0026, -0.0017] | 0.0000 | 0.3200 | 0.0021 | 0.1810 | practically tied to control |
| CIFAR DiagLap movement | scale=1.0e-03 | global | posterior-dense | 5 | 0.1447 | 0.1469 | -0.0021 | [-0.0026, -0.0017] | 0.0000 | 0.3200 | 0.0021 | 0.1810 | practically tied to control |
| CIFAR DiagLap movement | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1447 | 0.1787 | -0.0340 | [-0.0361, -0.0318] | 0.0000 | 0.0000 | 0.0340 | 0.9687 | control closer to ticket |
| CIFAR DiagLap movement | scale=1.0e-10 | global | posterior-random | 5 | 0.1469 | 0.0918 | 0.0551 | [0.0530, 0.0572] | 1.0000 | 0.0000 | 0.0551 | 0.8489 | posterior separates from random |
| CIFAR DiagLap movement | scale=1.0e-10 | global | posterior-chain | 5 | 0.1469 | 0.1469 | 2.82e-06 | [3.67e-07, 5.27e-06] | 1.0000 | 1.0000 | 2.82e-06 | 6.69e-07 | practically tied to control |
| CIFAR DiagLap movement | scale=1.0e-10 | global | posterior-dense | 5 | 0.1469 | 0.1469 | 2.82e-06 | [3.67e-07, 5.27e-06] | 1.0000 | 1.0000 | 2.82e-06 | 6.69e-07 | practically tied to control |
| CIFAR DiagLap movement | scale=1.0e-10 | global | posterior-rewind | 5 | 0.1469 | 0.1787 | -0.0318 | [-0.0342, -0.0294] | 0.0000 | 0.0000 | 0.0318 | 0.9856 | control closer to ticket |
| CIFAR DiagLap movement | scale=3.0e-03 | global | posterior-random | 5 | 0.1400 | 0.0918 | 0.0482 | [0.0462, 0.0502] | 1.0000 | 0.0000 | 0.0482 | 0.8530 | posterior separates from random |
| CIFAR DiagLap movement | scale=3.0e-03 | global | posterior-chain | 5 | 0.1400 | 0.1469 | -0.0069 | [-0.0078, -0.0060] | 0.0000 | 0.0000 | 0.0069 | 0.9104 | control closer to ticket |
| CIFAR DiagLap movement | scale=3.0e-03 | global | posterior-dense | 5 | 0.1400 | 0.1469 | -0.0069 | [-0.0078, -0.0060] | 0.0000 | 0.0000 | 0.0069 | 0.9104 | control closer to ticket |
| CIFAR DiagLap movement | scale=3.0e-03 | global | posterior-rewind | 5 | 0.1400 | 0.1787 | -0.0387 | [-0.0407, -0.0367] | 0.0000 | 0.0000 | 0.0387 | 0.9353 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e+00 | global | posterior-random | 5 | 0.1464 | 0.0917 | 0.0547 | [0.0520, 0.0573] | 1.0000 | 0.0000 | 0.0547 | 0.8616 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e+00 | global | posterior-chain | 5 | 0.1464 | 0.1458 | 0.0005 | [0.0004, 0.0007] | 1.0000 | 1.0000 | 0.0005 | 0.0076 | practically tied to control |
| CIFAR HeadLap | scale=1.0e+00 | global | posterior-rewind | 5 | 0.1464 | 0.1782 | -0.0318 | [-0.0338, -0.0298] | 0.0000 | 0.0000 | 0.0318 | 1.0027 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e+00 | head | posterior-random | 5 | 0.6716 | 0.6740 | -0.0024 | [-0.0052, 0.0003] | 0.2000 | 1.0000 | 0.0025 | 0.0093 | no random separation |
| CIFAR HeadLap | scale=1.0e+00 | head | posterior-chain | 5 | 0.6716 | 0.7068 | -0.0352 | [-0.0515, -0.0189] | 0.0000 | 0.0800 | 0.0352 | 0.5615 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e+00 | head | posterior-rewind | 5 | 0.6716 | 0.7191 | -0.0475 | [-0.0550, -0.0401] | 0.0000 | 0.0000 | 0.0475 | 1.0754 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-02 | global | posterior-random | 5 | 0.1459 | 0.0917 | 0.0542 | [0.0515, 0.0570] | 1.0000 | 0.0000 | 0.0542 | 0.8633 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-02 | global | posterior-chain | 5 | 0.1459 | 0.1458 | 0.0001 | [-2.13e-06, 0.0002] | 0.8000 | 1.0000 | 0.0001 | 0.0004 | practically tied to control |
| CIFAR HeadLap | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1459 | 0.1782 | -0.0323 | [-0.0343, -0.0302] | 0.0000 | 0.0000 | 0.0323 | 1.0052 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-02 | head | posterior-random | 5 | 0.6784 | 0.6741 | 0.0044 | [0.0002, 0.0085] | 0.8000 | 0.8200 | 0.0049 | 0.0478 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-02 | head | posterior-chain | 5 | 0.6784 | 0.7068 | -0.0284 | [-0.0443, -0.0124] | 0.0000 | 0.0800 | 0.0284 | 0.4373 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-02 | head | posterior-rewind | 5 | 0.6784 | 0.7191 | -0.0407 | [-0.0473, -0.0342] | 0.0000 | 0.0000 | 0.0407 | 1.0198 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-03 | global | posterior-random | 5 | 0.1458 | 0.0917 | 0.0541 | [0.0513, 0.0568] | 1.0000 | 0.0000 | 0.0541 | 0.8637 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-03 | global | posterior-chain | 5 | 0.1458 | 0.1458 | -4.90e-05 | [-0.0001, 4.29e-05] | 0.2000 | 1.0000 | 9.32e-05 | 0.0002 | practically tied to control |
| CIFAR HeadLap | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1458 | 0.1782 | -0.0324 | [-0.0345, -0.0304] | 0.0000 | 0.0000 | 0.0324 | 1.0052 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-03 | head | posterior-random | 5 | 0.6983 | 0.6741 | 0.0242 | [0.0177, 0.0307] | 1.0000 | 0.0800 | 0.0242 | 0.6632 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-03 | head | posterior-chain | 5 | 0.6983 | 0.7068 | -0.0085 | [-0.0231, 0.0061] | 0.4000 | 0.8200 | 0.0125 | 0.1790 | mixed |
| CIFAR HeadLap | scale=1.0e-03 | head | posterior-rewind | 5 | 0.6983 | 0.7191 | -0.0209 | [-0.0291, -0.0126] | 0.0000 | 0.0000 | 0.0209 | 0.4471 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-06 | global | posterior-random | 5 | 0.1458 | 0.0917 | 0.0541 | [0.0513, 0.0569] | 1.0000 | 0.0000 | 0.0541 | 0.8673 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-06 | global | posterior-chain | 5 | 0.1458 | 0.1458 | 5.88e-07 | [-1.75e-05, 1.87e-05] | 0.4000 | 1.0000 | 1.50e-05 | 1.41e-06 | practically tied to control |
| CIFAR HeadLap | scale=1.0e-06 | global | posterior-rewind | 5 | 0.1458 | 0.1782 | -0.0324 | [-0.0344, -0.0303] | 0.0000 | 0.0000 | 0.0324 | 1.0104 | control closer to ticket |
| CIFAR HeadLap | scale=1.0e-06 | head | posterior-random | 5 | 0.7067 | 0.6742 | 0.0326 | [0.0135, 0.0516] | 1.0000 | 0.0800 | 0.0326 | 0.4928 | posterior separates from random |
| CIFAR HeadLap | scale=1.0e-06 | head | posterior-chain | 5 | 0.7067 | 0.7068 | -3.91e-05 | [-0.0012, 0.0011] | 0.6000 | 1.0000 | 0.0011 | 0.0008 | practically tied to control |
| CIFAR HeadLap | scale=1.0e-06 | head | posterior-rewind | 5 | 0.7067 | 0.7191 | -0.0124 | [-0.0347, 0.0099] | 0.2000 | 0.3200 | 0.0131 | 0.1556 | control closer to ticket |
| CIFAR Hess16SubHMC | step=3.0e-04 | global | posterior-random | 5 | 0.1468 | 0.0917 | 0.0551 | [0.0531, 0.0572] | 1.0000 | 0.0000 | 0.0551 | 0.8442 | posterior separates from random |
| CIFAR Hess16SubHMC | step=3.0e-04 | global | posterior-chain | 5 | 0.1468 | 0.1468 | -1.64e-05 | [-5.48e-05, 2.19e-05] | 0.4000 | 1.0000 | 3.75e-05 | 3.22e-05 | practically tied to control |
| CIFAR Hess16SubHMC | step=3.0e-04 | global | posterior-dense | 5 | 0.1468 | 0.1468 | -1.64e-05 | [-5.48e-05, 2.19e-05] | 0.4000 | 1.0000 | 3.75e-05 | 3.22e-05 | practically tied to control |
| CIFAR Hess16SubHMC | step=3.0e-04 | global | posterior-rewind | 5 | 0.1468 | 0.1799 | -0.0331 | [-0.0347, -0.0315] | 0.0000 | 0.0000 | 0.0331 | 0.9051 | control closer to ticket |
| CIFAR Hess32SubHMC | step=3.0e-04 | global | posterior-random | 5 | 0.1461 | 0.0917 | 0.0545 | [0.0512, 0.0577] | 1.0000 | 0.0000 | 0.0545 | 0.8770 | posterior separates from random |
| CIFAR Hess32SubHMC | step=3.0e-04 | global | posterior-chain | 5 | 0.1461 | 0.1461 | 3.55e-05 | [3.86e-06, 6.72e-05] | 0.8000 | 1.0000 | 3.75e-05 | 2.52e-05 | practically tied to control |
| CIFAR Hess32SubHMC | step=3.0e-04 | global | posterior-dense | 5 | 0.1461 | 0.1461 | 3.55e-05 | [3.86e-06, 6.72e-05] | 0.8000 | 1.0000 | 3.75e-05 | 2.52e-05 | practically tied to control |
| CIFAR Hess32SubHMC | step=3.0e-04 | global | posterior-rewind | 5 | 0.1461 | 0.1788 | -0.0326 | [-0.0353, -0.0300] | 0.0000 | 0.0000 | 0.0326 | 1.0870 | control closer to ticket |
| CIFAR HessSubHMC | step=3.0e-04 | global | posterior-random | 5 | 0.1471 | 0.0916 | 0.0555 | [0.0530, 0.0580] | 1.0000 | 0.0000 | 0.0555 | 0.8663 | posterior separates from random |
| CIFAR HessSubHMC | step=3.0e-04 | global | posterior-chain | 5 | 0.1471 | 0.1471 | 2.30e-06 | [-7.09e-06, 1.17e-05] | 0.4000 | 1.0000 | 8.10e-06 | 1.67e-06 | practically tied to control |
| CIFAR HessSubHMC | step=3.0e-04 | global | posterior-dense | 5 | 0.1471 | 0.1471 | 2.30e-06 | [-7.09e-06, 1.17e-05] | 0.4000 | 1.0000 | 8.10e-06 | 1.67e-06 | practically tied to control |
| CIFAR HessSubHMC | step=3.0e-04 | global | posterior-rewind | 5 | 0.1471 | 0.1810 | -0.0339 | [-0.0351, -0.0327] | 0.0000 | 0.0000 | 0.0339 | 0.9838 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1303 | 0.0917 | 0.0385 | [0.0371, 0.0400] | 1.0000 | 0.0000 | 0.0385 | 0.8326 | posterior separates from random |
| CIFAR KFACLap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1303 | 0.1456 | -0.0154 | [-0.0165, -0.0143] | 0.0000 | 0.0000 | 0.0154 | 1.0672 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1303 | 0.1456 | -0.0154 | [-0.0165, -0.0143] | 0.0000 | 0.0000 | 0.0154 | 1.0672 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1303 | 0.1775 | -0.0473 | [-0.0493, -0.0452] | 0.0000 | 0.0000 | 0.0473 | 0.9278 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-03 | global | posterior-random | 5 | 0.1441 | 0.0917 | 0.0524 | [0.0506, 0.0542] | 1.0000 | 0.0000 | 0.0524 | 0.8337 | posterior separates from random |
| CIFAR KFACLap movement | scale=1.0e-03 | global | posterior-chain | 5 | 0.1441 | 0.1456 | -0.0016 | [-0.0022, -0.0009] | 0.0000 | 0.3200 | 0.0016 | 0.0924 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-03 | global | posterior-dense | 5 | 0.1441 | 0.1456 | -0.0016 | [-0.0022, -0.0009] | 0.0000 | 0.3200 | 0.0016 | 0.0924 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1441 | 0.1775 | -0.0334 | [-0.0355, -0.0313] | 0.0000 | 0.0000 | 0.0334 | 1.0040 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-04 | global | posterior-random | 5 | 0.1456 | 0.0917 | 0.0539 | [0.0519, 0.0559] | 1.0000 | 0.0000 | 0.0539 | 0.8371 | posterior separates from random |
| CIFAR KFACLap movement | scale=1.0e-04 | global | posterior-chain | 5 | 0.1456 | 0.1456 | -2.16e-05 | [-0.0003, 0.0002] | 0.6000 | 1.0000 | 0.0002 | 0.0021 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-04 | global | posterior-dense | 5 | 0.1456 | 0.1456 | -2.16e-05 | [-0.0003, 0.0002] | 0.6000 | 1.0000 | 0.0002 | 0.0021 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-04 | global | posterior-rewind | 5 | 0.1456 | 0.1775 | -0.0319 | [-0.0341, -0.0296] | 0.0000 | 0.0000 | 0.0319 | 1.0021 | control closer to ticket |
| CIFAR KFACLap movement | scale=1.0e-10 | global | posterior-random | 5 | 0.1456 | 0.0917 | 0.0539 | [0.0517, 0.0561] | 1.0000 | 0.0000 | 0.0539 | 0.8377 | posterior separates from random |
| CIFAR KFACLap movement | scale=1.0e-10 | global | posterior-chain | 5 | 0.1456 | 0.1456 | 1.17e-06 | [-3.51e-06, 5.85e-06] | 0.4000 | 1.0000 | 4.04e-06 | 1.10e-06 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-10 | global | posterior-dense | 5 | 0.1456 | 0.1456 | 1.17e-06 | [-3.51e-06, 5.85e-06] | 0.4000 | 1.0000 | 4.04e-06 | 1.10e-06 | practically tied to control |
| CIFAR KFACLap movement | scale=1.0e-10 | global | posterior-rewind | 5 | 0.1456 | 0.1775 | -0.0319 | [-0.0342, -0.0296] | 0.0000 | 0.0000 | 0.0319 | 0.9916 | control closer to ticket |
| CIFAR LowRank128Lap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1351 | 0.0917 | 0.0434 | [0.0407, 0.0461] | 1.0000 | 0.0000 | 0.0434 | 0.9161 | posterior separates from random |
| CIFAR LowRank128Lap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1351 | 0.1453 | -0.0102 | [-0.0106, -0.0098] | 0.0000 | 0.0000 | 0.0102 | 1.0288 | control closer to ticket |
| CIFAR LowRank128Lap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1351 | 0.1453 | -0.0102 | [-0.0106, -0.0098] | 0.0000 | 0.0000 | 0.0102 | 1.0288 | control closer to ticket |
| CIFAR LowRank128Lap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1351 | 0.1780 | -0.0429 | [-0.0449, -0.0409] | 0.0000 | 0.0000 | 0.0429 | 0.9765 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1358 | 0.0918 | 0.0441 | [0.0420, 0.0461] | 1.0000 | 0.0000 | 0.0441 | 0.8794 | posterior separates from random |
| CIFAR LowRank32Lap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1358 | 0.1457 | -0.0098 | [-0.0112, -0.0085] | 0.0000 | 0.0000 | 0.0098 | 0.9466 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1358 | 0.1457 | -0.0098 | [-0.0112, -0.0085] | 0.0000 | 0.0000 | 0.0098 | 0.9466 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1358 | 0.1795 | -0.0436 | [-0.0446, -0.0427] | 0.0000 | 0.0000 | 0.0436 | 0.8752 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-03 | global | posterior-random | 5 | 0.1450 | 0.0918 | 0.0532 | [0.0503, 0.0561] | 1.0000 | 0.0000 | 0.0532 | 0.9153 | posterior separates from random |
| CIFAR LowRank32Lap movement | scale=1.0e-03 | global | posterior-chain | 5 | 0.1450 | 0.1457 | -0.0007 | [-0.0008, -0.0005] | 0.0000 | 0.8200 | 0.0007 | 0.0377 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=1.0e-03 | global | posterior-dense | 5 | 0.1450 | 0.1457 | -0.0007 | [-0.0008, -0.0005] | 0.0000 | 0.8200 | 0.0007 | 0.0377 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1450 | 0.1795 | -0.0345 | [-0.0362, -0.0328] | 0.0000 | 0.0000 | 0.0345 | 0.9262 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=1.0e-04 | global | posterior-random | 5 | 0.1456 | 0.0918 | 0.0538 | [0.0507, 0.0569] | 1.0000 | 0.0000 | 0.0538 | 0.9237 | posterior separates from random |
| CIFAR LowRank32Lap movement | scale=1.0e-04 | global | posterior-chain | 5 | 0.1456 | 0.1457 | -6.21e-05 | [-0.0002, 5.69e-05] | 0.4000 | 1.0000 | 0.0001 | 0.0008 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=1.0e-04 | global | posterior-dense | 5 | 0.1456 | 0.1457 | -6.21e-05 | [-0.0002, 5.69e-05] | 0.4000 | 1.0000 | 0.0001 | 0.0008 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=1.0e-04 | global | posterior-rewind | 5 | 0.1456 | 0.1795 | -0.0339 | [-0.0358, -0.0319] | 0.0000 | 0.0000 | 0.0339 | 0.9252 | control closer to ticket |
| CIFAR LowRank32Lap movement | scale=3.0e-03 | global | posterior-random | 5 | 0.1428 | 0.0918 | 0.0510 | [0.0485, 0.0536] | 1.0000 | 0.0000 | 0.0510 | 0.9024 | posterior separates from random |
| CIFAR LowRank32Lap movement | scale=3.0e-03 | global | posterior-chain | 5 | 0.1428 | 0.1457 | -0.0028 | [-0.0034, -0.0023] | 0.0000 | 0.3200 | 0.0028 | 0.2976 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=3.0e-03 | global | posterior-dense | 5 | 0.1428 | 0.1457 | -0.0028 | [-0.0034, -0.0023] | 0.0000 | 0.3200 | 0.0028 | 0.2976 | practically tied to control |
| CIFAR LowRank32Lap movement | scale=3.0e-03 | global | posterior-rewind | 5 | 0.1428 | 0.1795 | -0.0367 | [-0.0380, -0.0353] | 0.0000 | 0.0000 | 0.0367 | 0.9191 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1339 | 0.0918 | 0.0421 | [0.0391, 0.0451] | 1.0000 | 0.0000 | 0.0421 | 0.9287 | posterior separates from random |
| CIFAR LowRank64Lap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1339 | 0.1433 | -0.0095 | [-0.0099, -0.0091] | 0.0000 | 0.0000 | 0.0095 | 0.8770 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1339 | 0.1433 | -0.0095 | [-0.0099, -0.0091] | 0.0000 | 0.0000 | 0.0095 | 0.8770 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1339 | 0.1766 | -0.0427 | [-0.0448, -0.0406] | 0.0000 | 0.0000 | 0.0427 | 1.0352 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-03 | global | posterior-random | 5 | 0.1426 | 0.0918 | 0.0508 | [0.0478, 0.0539] | 1.0000 | 0.0000 | 0.0508 | 0.9049 | posterior separates from random |
| CIFAR LowRank64Lap movement | scale=1.0e-03 | global | posterior-chain | 5 | 0.1426 | 0.1433 | -0.0007 | [-0.0009, -0.0006] | 0.0000 | 0.8200 | 0.0007 | 0.0124 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=1.0e-03 | global | posterior-dense | 5 | 0.1426 | 0.1433 | -0.0007 | [-0.0009, -0.0006] | 0.0000 | 0.8200 | 0.0007 | 0.0124 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1426 | 0.1766 | -0.0340 | [-0.0360, -0.0320] | 0.0000 | 0.0000 | 0.0340 | 1.1125 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=1.0e-04 | global | posterior-random | 5 | 0.1432 | 0.0918 | 0.0514 | [0.0484, 0.0545] | 1.0000 | 0.0000 | 0.0514 | 0.9045 | posterior separates from random |
| CIFAR LowRank64Lap movement | scale=1.0e-04 | global | posterior-chain | 5 | 0.1432 | 0.1433 | -0.0001 | [-0.0003, 2.35e-05] | 0.2000 | 0.8200 | 0.0002 | 0.0013 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=1.0e-04 | global | posterior-dense | 5 | 0.1432 | 0.1433 | -0.0001 | [-0.0003, 2.35e-05] | 0.2000 | 0.8200 | 0.0002 | 0.0013 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=1.0e-04 | global | posterior-rewind | 5 | 0.1432 | 0.1766 | -0.0334 | [-0.0353, -0.0314] | 0.0000 | 0.0000 | 0.0334 | 1.1162 | control closer to ticket |
| CIFAR LowRank64Lap movement | scale=3.0e-03 | global | posterior-random | 5 | 0.1406 | 0.0918 | 0.0488 | [0.0459, 0.0518] | 1.0000 | 0.0000 | 0.0488 | 0.9051 | posterior separates from random |
| CIFAR LowRank64Lap movement | scale=3.0e-03 | global | posterior-chain | 5 | 0.1406 | 0.1433 | -0.0027 | [-0.0030, -0.0024] | 0.0000 | 0.8200 | 0.0027 | 0.1484 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=3.0e-03 | global | posterior-dense | 5 | 0.1406 | 0.1433 | -0.0027 | [-0.0030, -0.0024] | 0.0000 | 0.8200 | 0.0027 | 0.1484 | practically tied to control |
| CIFAR LowRank64Lap movement | scale=3.0e-03 | global | posterior-rewind | 5 | 0.1406 | 0.1766 | -0.0359 | [-0.0381, -0.0338] | 0.0000 | 0.0000 | 0.0359 | 1.0923 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-02 | global | posterior-random | 5 | 0.1351 | 0.0918 | 0.0433 | [0.0406, 0.0460] | 1.0000 | 0.0000 | 0.0433 | 0.8727 | posterior separates from random |
| CIFAR LowRankLap movement | scale=1.0e-02 | global | posterior-chain | 5 | 0.1351 | 0.1456 | -0.0105 | [-0.0110, -0.0100] | 0.0000 | 0.0000 | 0.0105 | 0.9711 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-02 | global | posterior-dense | 5 | 0.1351 | 0.1456 | -0.0105 | [-0.0110, -0.0100] | 0.0000 | 0.0000 | 0.0105 | 0.9711 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-02 | global | posterior-rewind | 5 | 0.1351 | 0.1777 | -0.0427 | [-0.0445, -0.0409] | 0.0000 | 0.0000 | 0.0427 | 0.9788 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-03 | global | posterior-random | 5 | 0.1447 | 0.0918 | 0.0529 | [0.0500, 0.0559] | 1.0000 | 0.0000 | 0.0529 | 0.8730 | posterior separates from random |
| CIFAR LowRankLap movement | scale=1.0e-03 | global | posterior-chain | 5 | 0.1447 | 0.1456 | -0.0008 | [-0.0010, -0.0007] | 0.0000 | 0.8200 | 0.0008 | 0.0157 | practically tied to control |
| CIFAR LowRankLap movement | scale=1.0e-03 | global | posterior-dense | 5 | 0.1447 | 0.1456 | -0.0008 | [-0.0010, -0.0007] | 0.0000 | 0.8200 | 0.0008 | 0.0157 | practically tied to control |
| CIFAR LowRankLap movement | scale=1.0e-03 | global | posterior-rewind | 5 | 0.1447 | 0.1777 | -0.0330 | [-0.0351, -0.0309] | 0.0000 | 0.0000 | 0.0330 | 1.0115 | control closer to ticket |
| CIFAR LowRankLap movement | scale=1.0e-04 | global | posterior-random | 5 | 0.1455 | 0.0918 | 0.0537 | [0.0508, 0.0566] | 1.0000 | 0.0000 | 0.0537 | 0.8750 | posterior separates from random |
| CIFAR LowRankLap movement | scale=1.0e-04 | global | posterior-chain | 5 | 0.1455 | 0.1456 | -9.26e-05 | [-0.0002, -2.26e-05] | 0.0000 | 0.8200 | 9.26e-05 | 0.0003 | practically tied to control |
| CIFAR LowRankLap movement | scale=1.0e-04 | global | posterior-dense | 5 | 0.1455 | 0.1456 | -9.26e-05 | [-0.0002, -2.26e-05] | 0.0000 | 0.8200 | 9.26e-05 | 0.0003 | practically tied to control |
| CIFAR LowRankLap movement | scale=1.0e-04 | global | posterior-rewind | 5 | 0.1455 | 0.1777 | -0.0323 | [-0.0344, -0.0302] | 0.0000 | 0.0000 | 0.0323 | 1.0173 | control closer to ticket |
| CIFAR LowRankLap movement | scale=3.0e-03 | global | posterior-random | 5 | 0.1424 | 0.0918 | 0.0506 | [0.0477, 0.0535] | 1.0000 | 0.0000 | 0.0506 | 0.8681 | posterior separates from random |
| CIFAR LowRankLap movement | scale=3.0e-03 | global | posterior-chain | 5 | 0.1424 | 0.1456 | -0.0032 | [-0.0034, -0.0029] | 0.0000 | 0.8200 | 0.0032 | 0.1852 | practically tied to control |
| CIFAR LowRankLap movement | scale=3.0e-03 | global | posterior-dense | 5 | 0.1424 | 0.1456 | -0.0032 | [-0.0034, -0.0029] | 0.0000 | 0.8200 | 0.0032 | 0.1852 | practically tied to control |
| CIFAR LowRankLap movement | scale=3.0e-03 | global | posterior-rewind | 5 | 0.1424 | 0.1777 | -0.0354 | [-0.0374, -0.0334] | 0.0000 | 0.0000 | 0.0354 | 0.9999 | control closer to ticket |
| CIFAR RandSubHMC | step=3.0e-03 | global | posterior-random | 5 | 0.1440 | 0.0918 | 0.0523 | [0.0488, 0.0557] | 1.0000 | 0.0000 | 0.0523 | 0.8343 | posterior separates from random |
| CIFAR RandSubHMC | step=3.0e-03 | global | posterior-chain | 5 | 0.1440 | 0.1440 | -2.23e-05 | [-9.56e-05, 5.11e-05] | 0.2000 | 0.8200 | 6.82e-05 | 0.0013 | practically tied to control |
| CIFAR RandSubHMC | step=3.0e-03 | global | posterior-dense | 5 | 0.1440 | 0.1440 | -2.23e-05 | [-9.56e-05, 5.11e-05] | 0.2000 | 0.8200 | 6.82e-05 | 0.0013 | practically tied to control |
| CIFAR RandSubHMC | step=3.0e-03 | global | posterior-rewind | 5 | 0.1440 | 0.1779 | -0.0339 | [-0.0377, -0.0301] | 0.0000 | 0.0000 | 0.0339 | 0.9741 | control closer to ticket |
| CIFAR SGHMC movement | lr=1.0e-07 | global | posterior-random | 5 | 0.1419 | 0.0918 | 0.0501 | [0.0474, 0.0529] | 1.0000 | 0.0000 | 0.0501 | 0.8869 | posterior separates from random |
| CIFAR SGHMC movement | lr=1.0e-07 | global | posterior-chain | 5 | 0.1419 | 0.1457 | -0.0037 | [-0.0044, -0.0031] | 0.0000 | 0.3200 | 0.0037 | 0.2216 | practically tied to control |
| CIFAR SGHMC movement | lr=1.0e-07 | global | posterior-dense | 5 | 0.1419 | 0.1457 | -0.0037 | [-0.0044, -0.0031] | 0.0000 | 0.3200 | 0.0037 | 0.2216 | practically tied to control |
| CIFAR SGHMC movement | lr=1.0e-07 | global | posterior-rewind | 5 | 0.1419 | 0.1777 | -0.0358 | [-0.0384, -0.0332] | 0.0000 | 0.0000 | 0.0358 | 1.0436 | control closer to ticket |
| CIFAR SGHMC movement | lr=1.0e-10 | global | posterior-random | 5 | 0.1456 | 0.0918 | 0.0539 | [0.0506, 0.0571] | 1.0000 | 0.0000 | 0.0539 | 0.8915 | posterior separates from random |
| CIFAR SGHMC movement | lr=1.0e-10 | global | posterior-chain | 5 | 0.1456 | 0.1457 | -4.65e-05 | [-0.0001, 4.90e-05] | 0.2000 | 0.8200 | 0.0001 | 0.0003 | practically tied to control |
| CIFAR SGHMC movement | lr=1.0e-10 | global | posterior-dense | 5 | 0.1456 | 0.1457 | -4.65e-05 | [-0.0001, 4.90e-05] | 0.2000 | 0.8200 | 0.0001 | 0.0003 | practically tied to control |
| CIFAR SGHMC movement | lr=1.0e-10 | global | posterior-rewind | 5 | 0.1456 | 0.1777 | -0.0321 | [-0.0349, -0.0293] | 0.0000 | 0.0000 | 0.0321 | 1.1028 | control closer to ticket |
| CIFAR SGHMC movement | lr=3.0e-07 | global | posterior-random | 5 | 0.1360 | 0.0918 | 0.0443 | [0.0420, 0.0465] | 1.0000 | 0.0000 | 0.0443 | 0.8740 | posterior separates from random |
| CIFAR SGHMC movement | lr=3.0e-07 | global | posterior-chain | 5 | 0.1360 | 0.1457 | -0.0096 | [-0.0106, -0.0087] | 0.0000 | 0.0000 | 0.0096 | 0.9367 | control closer to ticket |
| CIFAR SGHMC movement | lr=3.0e-07 | global | posterior-dense | 5 | 0.1360 | 0.1457 | -0.0096 | [-0.0106, -0.0087] | 0.0000 | 0.0000 | 0.0096 | 0.9367 | control closer to ticket |
| CIFAR SGHMC movement | lr=3.0e-07 | global | posterior-rewind | 5 | 0.1360 | 0.1777 | -0.0417 | [-0.0444, -0.0390] | 0.0000 | 0.0000 | 0.0417 | 1.0167 | control closer to ticket |
| CIFAR SGHMC movement | lr=3.0e-08 | global | posterior-random | 5 | 0.1445 | 0.0917 | 0.0528 | [0.0499, 0.0556] | 1.0000 | 0.0000 | 0.0528 | 0.8835 | posterior separates from random |
| CIFAR SGHMC movement | lr=3.0e-08 | global | posterior-chain | 5 | 0.1445 | 0.1457 | -0.0011 | [-0.0016, -0.0006] | 0.0000 | 0.8200 | 0.0011 | 0.0268 | practically tied to control |
| CIFAR SGHMC movement | lr=3.0e-08 | global | posterior-dense | 5 | 0.1445 | 0.1457 | -0.0011 | [-0.0016, -0.0006] | 0.0000 | 0.8200 | 0.0011 | 0.0268 | practically tied to control |
| CIFAR SGHMC movement | lr=3.0e-08 | global | posterior-rewind | 5 | 0.1445 | 0.1777 | -0.0332 | [-0.0362, -0.0302] | 0.0000 | 0.0000 | 0.0332 | 1.0678 | control closer to ticket |
| CIFAR SGLD movement | lr=1.0e-06 | global | posterior-random | 5 | 0.1425 | 0.0918 | 0.0507 | [0.0483, 0.0531] | 1.0000 | 0.0000 | 0.0507 | 0.8416 | posterior separates from random |
| CIFAR SGLD movement | lr=1.0e-06 | global | posterior-chain | 5 | 0.1425 | 0.1441 | -0.0017 | [-0.0022, -0.0011] | 0.0000 | 0.8200 | 0.0017 | 0.1223 | practically tied to control |
| CIFAR SGLD movement | lr=1.0e-06 | global | posterior-dense | 5 | 0.1425 | 0.1441 | -0.0017 | [-0.0022, -0.0011] | 0.0000 | 0.8200 | 0.0017 | 0.1223 | practically tied to control |
| CIFAR SGLD movement | lr=1.0e-06 | global | posterior-rewind | 5 | 0.1425 | 0.1784 | -0.0359 | [-0.0378, -0.0341] | 0.0000 | 0.0000 | 0.0359 | 0.9628 | control closer to ticket |
| CIFAR SGLD movement | lr=1.0e-10 | global | posterior-random | 5 | 0.1441 | 0.0918 | 0.0524 | [0.0500, 0.0548] | 1.0000 | 0.0000 | 0.0524 | 0.8573 | posterior separates from random |
| CIFAR SGLD movement | lr=1.0e-10 | global | posterior-chain | 5 | 0.1441 | 0.1441 | -1.75e-05 | [-5.45e-05, 1.95e-05] | 0.2000 | 1.0000 | 3.42e-05 | 3.26e-05 | practically tied to control |
| CIFAR SGLD movement | lr=1.0e-10 | global | posterior-dense | 5 | 0.1441 | 0.1441 | -1.75e-05 | [-5.45e-05, 1.95e-05] | 0.2000 | 1.0000 | 3.42e-05 | 3.26e-05 | practically tied to control |
| CIFAR SGLD movement | lr=1.0e-10 | global | posterior-rewind | 5 | 0.1441 | 0.1784 | -0.0343 | [-0.0361, -0.0324] | 0.0000 | 0.0000 | 0.0343 | 0.9599 | control closer to ticket |
| CIFAR SGLD movement | lr=3.0e-06 | global | posterior-random | 5 | 0.1381 | 0.0918 | 0.0463 | [0.0437, 0.0489] | 1.0000 | 0.0000 | 0.0463 | 0.8607 | posterior separates from random |
| CIFAR SGLD movement | lr=3.0e-06 | global | posterior-chain | 5 | 0.1381 | 0.1441 | -0.0061 | [-0.0067, -0.0055] | 0.0000 | 0.0800 | 0.0061 | 0.6541 | control closer to ticket |
| CIFAR SGLD movement | lr=3.0e-06 | global | posterior-dense | 5 | 0.1381 | 0.1441 | -0.0061 | [-0.0067, -0.0055] | 0.0000 | 0.0800 | 0.0061 | 0.6541 | control closer to ticket |
| CIFAR SGLD movement | lr=3.0e-06 | global | posterior-rewind | 5 | 0.1381 | 0.1784 | -0.0403 | [-0.0419, -0.0387] | 0.0000 | 0.0000 | 0.0403 | 0.9411 | control closer to ticket |
| CIFAR SGLD-3chain | 30ep rewind r5 p0.30 | global | posterior-random | 5 | 0.1368 | 0.0918 | 0.0450 | [0.0428, 0.0473] | 1.0000 | 0.0000 | 0.0450 | 0.8941 | posterior separates from random |
| CIFAR SGLD-3chain | 30ep rewind r5 p0.30 | global | posterior-chain | 5 | 0.1368 | 0.1368 | -4.88e-06 | [-4.86e-05, 3.89e-05] | 0.4000 | 1.0000 | 3.37e-05 | 6.31e-05 | practically tied to control |
| CIFAR SGLD-3chain | 30ep rewind r5 p0.30 | global | posterior-dense | 5 | 0.1368 | 0.1460 | -0.0092 | [-0.0111, -0.0072] | 0.0000 | 0.0000 | 0.0092 | 1.1503 | control closer to ticket |
| CIFAR SGLD-3chain | 30ep rewind r5 p0.30 | global | posterior-rewind | 5 | 0.1368 | 0.1800 | -0.0432 | [-0.0457, -0.0407] | 0.0000 | 0.0000 | 0.0432 | 0.9115 | control closer to ticket |
| CIFAR SWAG | 30ep rewind r5 p0.30 | global | posterior-random | 5 | 0.1361 | 0.0918 | 0.0443 | [0.0421, 0.0464] | 1.0000 | 0.0000 | 0.0443 | 0.9106 | posterior separates from random |
| CIFAR SWAG | 30ep rewind r5 p0.30 | global | posterior-chain | 5 | 0.1361 | 0.1361 | -5.38e-05 | [-0.0003, 0.0002] | 0.6000 | 0.8200 | 0.0003 | 0.0227 | practically tied to control |
| CIFAR SWAG | 30ep rewind r5 p0.30 | global | posterior-dense | 5 | 0.1361 | 0.1463 | -0.0102 | [-0.0122, -0.0082] | 0.0000 | 0.0000 | 0.0102 | 1.1195 | control closer to ticket |
| CIFAR SWAG | 30ep rewind r5 p0.30 | global | posterior-rewind | 5 | 0.1361 | 0.1786 | -0.0426 | [-0.0440, -0.0412] | 0.0000 | 0.0000 | 0.0426 | 0.9953 | control closer to ticket |
| CIFAR SWAG20 movement | scale=1.0e+00 | global | posterior-random | 5 | 0.1455 | 0.0917 | 0.0537 | [0.0506, 0.0569] | 1.0000 | 0.0000 | 0.0537 | 0.8997 | posterior separates from random |
| CIFAR SWAG20 movement | scale=1.0e+00 | global | posterior-chain | 5 | 0.1455 | 0.1455 | -1.16e-05 | [-0.0001, 0.0001] | 0.6000 | 1.0000 | 0.0001 | 0.0003 | practically tied to control |
| CIFAR SWAG20 movement | scale=1.0e+00 | global | posterior-dense | 5 | 0.1455 | 0.1455 | -1.16e-05 | [-0.0001, 0.0001] | 0.6000 | 1.0000 | 0.0001 | 0.0003 | practically tied to control |
| CIFAR SWAG20 movement | scale=1.0e+00 | global | posterior-rewind | 5 | 0.1455 | 0.1782 | -0.0328 | [-0.0342, -0.0313] | 0.0000 | 0.0000 | 0.0328 | 1.0212 | control closer to ticket |
| CIFAR SWAG20 movement | scale=1.6e+01 | global | posterior-random | 5 | 0.1454 | 0.0917 | 0.0537 | [0.0504, 0.0570] | 1.0000 | 0.0000 | 0.0537 | 0.8995 | posterior separates from random |
| CIFAR SWAG20 movement | scale=1.6e+01 | global | posterior-chain | 5 | 0.1454 | 0.1455 | -7.02e-05 | [-0.0002, 5.69e-05] | 0.2000 | 1.0000 | 0.0001 | 0.0002 | practically tied to control |
| CIFAR SWAG20 movement | scale=1.6e+01 | global | posterior-dense | 5 | 0.1454 | 0.1455 | -7.02e-05 | [-0.0002, 5.69e-05] | 0.2000 | 1.0000 | 0.0001 | 0.0002 | practically tied to control |
| CIFAR SWAG20 movement | scale=1.6e+01 | global | posterior-rewind | 5 | 0.1454 | 0.1782 | -0.0328 | [-0.0343, -0.0313] | 0.0000 | 0.0000 | 0.0328 | 1.0233 | control closer to ticket |
| CIFAR SWAG20 movement | scale=6.4e+01 | global | posterior-random | 5 | 0.1453 | 0.0917 | 0.0536 | [0.0504, 0.0568] | 1.0000 | 0.0000 | 0.0536 | 0.8932 | posterior separates from random |
| CIFAR SWAG20 movement | scale=6.4e+01 | global | posterior-chain | 5 | 0.1453 | 0.1455 | -0.0002 | [-0.0004, 6.54e-05] | 0.2000 | 1.0000 | 0.0003 | 0.0013 | practically tied to control |
| CIFAR SWAG20 movement | scale=6.4e+01 | global | posterior-dense | 5 | 0.1453 | 0.1455 | -0.0002 | [-0.0004, 6.54e-05] | 0.2000 | 1.0000 | 0.0003 | 0.0013 | practically tied to control |
| CIFAR SWAG20 movement | scale=6.4e+01 | global | posterior-rewind | 5 | 0.1453 | 0.1782 | -0.0329 | [-0.0344, -0.0314] | 0.0000 | 0.0000 | 0.0329 | 1.0271 | control closer to ticket |
| CIFAR TrajSubHMC | step=1.0e-03 | global | posterior-random | 5 | 0.2290 | 0.0917 | 0.1373 | [0.1297, 0.1449] | 1.0000 | 0.0000 | 0.1373 | 0.9335 | posterior separates from random |
| CIFAR TrajSubHMC | step=1.0e-03 | global | posterior-chain | 5 | 0.2290 | 0.2292 | -0.0002 | [-0.0003, -3.55e-05] | 0.4000 | 1.0000 | 0.0002 | 0.0033 | practically tied to control |
| CIFAR TrajSubHMC | step=1.0e-03 | global | posterior-dense | 5 | 0.2290 | 0.2292 | -0.0002 | [-0.0003, -3.55e-05] | 0.4000 | 1.0000 | 0.0002 | 0.0033 | practically tied to control |
| CIFAR TrajSubHMC | step=1.0e-03 | global | posterior-rewind | 5 | 0.2290 | 0.1792 | 0.0498 | [0.0449, 0.0548] | 1.0000 | 0.0000 | 0.0498 | 1.2575 | posterior closer to ticket |
| CIFAR TrajSubHMC | step=3.0e-04 | global | posterior-random | 5 | 0.2291 | 0.0916 | 0.1374 | [0.1298, 0.1451] | 1.0000 | 0.0000 | 0.1374 | 0.9343 | posterior separates from random |
| CIFAR TrajSubHMC | step=3.0e-04 | global | posterior-chain | 5 | 0.2291 | 0.2292 | -0.0001 | [-0.0001, -6.10e-05] | 0.0000 | 1.0000 | 0.0001 | 0.0005 | practically tied to control |
| CIFAR TrajSubHMC | step=3.0e-04 | global | posterior-dense | 5 | 0.2291 | 0.2292 | -0.0001 | [-0.0001, -6.10e-05] | 0.0000 | 1.0000 | 0.0001 | 0.0005 | practically tied to control |
| CIFAR TrajSubHMC | step=3.0e-04 | global | posterior-rewind | 5 | 0.2291 | 0.1792 | 0.0499 | [0.0449, 0.0550] | 1.0000 | 0.0000 | 0.0499 | 1.2599 | posterior closer to ticket |
| CIFAR cSGLD movement | lr=1.0e-05 | global | posterior-random | 5 | 0.1260 | 0.0918 | 0.0342 | [0.0335, 0.0350] | 1.0000 | 0.0000 | 0.0342 | 0.8274 | posterior separates from random |
| CIFAR cSGLD movement | lr=1.0e-05 | global | posterior-chain | 5 | 0.1260 | 0.1454 | -0.0194 | [-0.0203, -0.0186] | 0.0000 | 0.0000 | 0.0194 | 0.9612 | control closer to ticket |
| CIFAR cSGLD movement | lr=1.0e-05 | global | posterior-dense | 5 | 0.1260 | 0.1454 | -0.0194 | [-0.0203, -0.0186] | 0.0000 | 0.0000 | 0.0194 | 0.9612 | control closer to ticket |
| CIFAR cSGLD movement | lr=1.0e-05 | global | posterior-rewind | 5 | 0.1260 | 0.1789 | -0.0529 | [-0.0552, -0.0506] | 0.0000 | 0.0000 | 0.0529 | 0.8839 | control closer to ticket |
| CIFAR cSGLD movement | lr=1.0e-06 | global | posterior-random | 5 | 0.1422 | 0.0918 | 0.0504 | [0.0492, 0.0516] | 1.0000 | 0.0000 | 0.0504 | 0.8387 | posterior separates from random |
| CIFAR cSGLD movement | lr=1.0e-06 | global | posterior-chain | 5 | 0.1422 | 0.1454 | -0.0033 | [-0.0036, -0.0030] | 0.0000 | 0.0800 | 0.0033 | 0.5894 | practically tied to control |
| CIFAR cSGLD movement | lr=1.0e-06 | global | posterior-dense | 5 | 0.1422 | 0.1454 | -0.0033 | [-0.0036, -0.0030] | 0.0000 | 0.0800 | 0.0033 | 0.5894 | practically tied to control |
| CIFAR cSGLD movement | lr=1.0e-06 | global | posterior-rewind | 5 | 0.1422 | 0.1789 | -0.0368 | [-0.0388, -0.0348] | 0.0000 | 0.0000 | 0.0368 | 0.9128 | control closer to ticket |
| CIFAR cSGLD movement | lr=1.0e-10 | global | posterior-random | 5 | 0.1454 | 0.0918 | 0.0537 | [0.0522, 0.0552] | 1.0000 | 0.0000 | 0.0537 | 0.8465 | posterior separates from random |
| CIFAR cSGLD movement | lr=1.0e-10 | global | posterior-chain | 5 | 0.1454 | 0.1454 | 7.77e-06 | [-3.52e-05, 5.07e-05] | 0.6000 | 1.0000 | 3.61e-05 | 2.83e-05 | practically tied to control |
| CIFAR cSGLD movement | lr=1.0e-10 | global | posterior-dense | 5 | 0.1454 | 0.1454 | 7.77e-06 | [-3.52e-05, 5.07e-05] | 0.6000 | 1.0000 | 3.61e-05 | 2.83e-05 | practically tied to control |
| CIFAR cSGLD movement | lr=1.0e-10 | global | posterior-rewind | 5 | 0.1454 | 0.1789 | -0.0335 | [-0.0354, -0.0316] | 0.0000 | 0.0000 | 0.0335 | 0.9367 | control closer to ticket |
| CIFAR cSGLD movement | lr=3.0e-06 | global | posterior-random | 5 | 0.1371 | 0.0918 | 0.0454 | [0.0437, 0.0471] | 1.0000 | 0.0000 | 0.0454 | 0.8689 | posterior separates from random |
| CIFAR cSGLD movement | lr=3.0e-06 | global | posterior-chain | 5 | 0.1371 | 0.1454 | -0.0083 | [-0.0085, -0.0081] | 0.0000 | 0.0000 | 0.0083 | 1.1315 | control closer to ticket |
| CIFAR cSGLD movement | lr=3.0e-06 | global | posterior-dense | 5 | 0.1371 | 0.1454 | -0.0083 | [-0.0085, -0.0081] | 0.0000 | 0.0000 | 0.0083 | 1.1315 | control closer to ticket |
| CIFAR cSGLD movement | lr=3.0e-06 | global | posterior-rewind | 5 | 0.1371 | 0.1789 | -0.0418 | [-0.0438, -0.0398] | 0.0000 | 0.0000 | 0.0418 | 0.9150 | control closer to ticket |
| Fashion Gate1 | r5 p0.30 | global | posterior-random | 5 | 0.2114 | 0.0917 | 0.1197 | [0.1177, 0.1218] | 1.0000 | 0.0000 | 0.1197 | 0.8053 | posterior separates from random |
| Fashion Gate1 | r5 p0.30 | global | posterior-chain | 5 | 0.2114 | 0.2122 | -0.0008 | [-0.0009, -0.0006] | 0.0000 | 0.3200 | 0.0008 | 0.0338 | practically tied to control |
| Fashion Gate1 | r5 p0.30 | global | posterior-dense | 5 | 0.2114 | 0.5255 | -0.3141 | [-0.3482, -0.2800] | 0.0000 | 0.0000 | 0.3141 | 1.0165 | control closer to ticket |
| MNIST Gate1 | r5 p0.30 | global | posterior-random | 5 | 0.2373 | 0.0918 | 0.1455 | [0.1446, 0.1465] | 1.0000 | 0.0000 | 0.1455 | 0.7996 | posterior separates from random |
| MNIST Gate1 | r5 p0.30 | global | posterior-chain | 5 | 0.2373 | 0.2385 | -0.0011 | [-0.0013, -0.0010] | 0.0000 | 0.3200 | 0.0011 | 0.2064 | practically tied to control |
| MNIST Gate1 | r5 p0.30 | global | posterior-dense | 5 | 0.2373 | 0.5924 | -0.3551 | [-0.3653, -0.3449] | 0.0000 | 0.0000 | 0.3551 | 0.8240 | control closer to ticket |

Generated by `scripts/run_mode_distribution_equivalence_audit.py`.
