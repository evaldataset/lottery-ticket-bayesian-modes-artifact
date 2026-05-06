# CIFAR-10 ResNet-20 Short SGLD Movement Sweep

The first two grids below are one-seed tuning diagnostics that train the dense
and IMP models once per grid and then vary SGLD step size from the same dense
checkpoint. A follow-up selected grid repeats the most informative step sizes
across five seeds.

## Five-Seed Selected Grid

| SGLD LR | Runs | Dense Acc | IMP Acc | Posterior | Random | Chain Start | Post-Chain Start | Post-Chain | Dense Mag | Sample Acc | State Clusters | Function Clusters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-10 | 5 | 0.8218 | 0.8647 | 0.1730 | 0.0918 | 0.1729 | 0.0000 | 0.9963 | 0.1729 | 0.8225 | 3.0 | 2.2 |
| 1e-6 | 5 | 0.8218 | 0.8647 | 0.1683 | 0.0917 | 0.1729 | -0.0046 | 0.6932 | 0.1729 | 0.8169 | 3.0 | 1.0 |
| 3e-6 | 5 | 0.8218 | 0.8647 | 0.1602 | 0.0917 | 0.1729 | -0.0127 | 0.5440 | 0.1729 | 0.7954 | 3.0 | 1.0 |

The selected five-seed result strengthens the one-seed diagnostic. SGLD support
can move substantially away from the dense chain start, but movement is not
ticket-directed: posterior-to-IMP overlap decreases as post-chain overlap
decreases.

## One-Seed Step-Size Tuning Grid

| SGLD LR | Dense Acc | IMP Acc | Posterior | Random | Chain Start | Post-Chain Start | Post-Chain | Dense Mag | Sample Acc | State Clusters | Function Clusters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-10 | 0.8233 | 0.8625 | 0.1701 | 0.0917 | 0.1701 | -0.0000 | 0.9965 | 0.1701 | 0.8243 | 3.0 | 3.0 |
| 3e-10 | 0.8233 | 0.8625 | 0.1703 | 0.0916 | 0.1701 | 0.0001 | 0.9933 | 0.1701 | 0.8243 | 3.0 | 2.0 |
| 1e-9 | 0.8233 | 0.8625 | 0.1703 | 0.0916 | 0.1701 | 0.0001 | 0.9887 | 0.1701 | 0.8242 | 3.0 | 2.0 |
| 3e-9 | 0.8233 | 0.8625 | 0.1704 | 0.0916 | 0.1701 | 0.0003 | 0.9801 | 0.1701 | 0.8242 | 3.0 | 3.0 |
| 1e-8 | 0.8233 | 0.8625 | 0.1701 | 0.0916 | 0.1701 | -0.0000 | 0.9635 | 0.1701 | 0.8245 | 3.0 | 4.0 |
| 3e-8 | 0.8233 | 0.8625 | 0.1701 | 0.0916 | 0.1701 | -0.0000 | 0.9383 | 0.1701 | 0.8241 | 3.0 | 1.0 |
| 1e-7 | 0.8235 | 0.8665 | 0.1670 | 0.0918 | 0.1679 | -0.0008 | 0.8880 | 0.1679 | 0.8253 | 3.0 | 1.0 |
| 3e-7 | 0.8235 | 0.8665 | 0.1659 | 0.0918 | 0.1679 | -0.0020 | 0.8150 | 0.1679 | 0.8239 | 3.0 | 1.0 |
| 1e-6 | 0.8235 | 0.8665 | 0.1619 | 0.0918 | 0.1679 | -0.0060 | 0.6949 | 0.1679 | 0.8188 | 3.0 | 1.0 |
| 3e-6 | 0.8235 | 0.8665 | 0.1547 | 0.0918 | 0.1679 | -0.0132 | 0.5437 | 0.1679 | 0.7975 | 3.0 | 2.0 |

Interpretation:

Increasing SGLD step size moves CIFAR supports away from the dense chain start
in both the tuning and selected grids. The repeated-seed selected grid gives the
cleaner evidence: at `1e-6`, posterior-to-chain-start overlap falls to `0.6932`
while sample accuracy remains usable at `0.8169`; at `3e-6`, overlap falls to
`0.5440`, but sample accuracy drops to `0.7954`. This movement does not make the
support more ticket-like: posterior-to-IMP overlap decreases below the dense
chain-start magnitude control as step size increases. The short CIFAR result
therefore matches the MNIST and HMC movement story: moving the posterior support
away from the starting checkpoint does not move it toward IMP.

Source files:

- `runs/cifar10_resnet20_sgld_movement_short_r5_p0p3/20260503_184103/metrics.json`
- `runs/cifar10_resnet20_sgld_movement_short_highlr_r5_p0p3/20260503_184406/metrics.json`
- `docs/cifar10_resnet20_sgld_movement_short_selected_r5_p0p3.md`
