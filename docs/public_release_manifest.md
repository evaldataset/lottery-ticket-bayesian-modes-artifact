# Public Release Manifest

Date: 2026-05-06

This manifest defines the minimum local package needed to inspect the
paper, regenerate generated statistics from included run artifacts,
and verify the core claims. The full per-file SHA256 inventory is in
`runs/public_release_manifest.json`.

`docs/external_validation_receipts.json` is intentionally excluded:
it is a mutable post-release registry for public URLs, source commits,
CI runs, and GPU logs.

Total files: 1863
Total bytes: 131.6 MiB

## Category Summary

| Category | Files | Bytes |
| --- | ---: | ---: |
| ci | 1 | 1.3 KiB |
| docs | 203 | 1.1 MiB |
| paper | 8 | 950.6 KiB |
| paper-figures | 6 | 220.4 KiB |
| paper-tables | 1 | 31.8 KiB |
| root | 12 | 159.5 KiB |
| run-artifacts | 1532 | 127.4 MiB |
| scripts | 78 | 1.6 MiB |
| source | 22 | 121.2 KiB |

## Required Artifacts

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| .dockerignore | 391 | `dc31f24c9cb52a69d1aefc300d7fcb8b9d7e2768a160b38c2f372ab23a56abc4` |
| .github/workflows/check.yml | 1311 | `8ce84a9c640c4ee066c10b0d45a92f10541190ad574c7f738d188693a189fd1f` |
| .gitignore | 175 | `472922ca7d9d71ec20d0083d5632066849555d76d60fcb2b8f9a0f72b4ed6ca9` |
| Dockerfile | 1293 | `0841b7b00b5e269d6e7bef5a5cc346b144138efa94f14d20cc8ae5d55f0b0105` |
| Dockerfile.gpu | 1672 | `c1b9fa5354b4d085ef3dd4cdda9033b770ee769c2a09c07940af0e9cd0b336ec` |
| LICENSE | 1074 | `1d0dde41ece0f9620937f1a81a67e295ace0eae180c009d862006e8f3b288458` |
| Makefile | 14578 | `86cc2912b8432e3c21d8c88589f5222920929afaff31aaba5232bc162e31d3fd` |
| README.md | 123495 | `8a9e5c5064eac1d0b64897534661420dc2065dc1ccc8a852156f2b243dd4ae33` |
| requirements-ci.txt | 82 | `ce5460165123c8f91ea7ba49296c66d53ae24b04b4ffd638184b4022616001e2` |
| requirements-gpu-lock.txt | 374 | `5d80d0caa3599bd900afdf0ea77837391f349553df4d1751720752980a2ddab2` |
| requirements-lock.txt | 425 | `373f546c4fefb1ca56a6cd1b92a3fc68689a0196aa993a4bc401e5166b8ae8f3` |
| docs/container_lock.md | 3747 | `bd98fc70ee38f013027243e6a87b1307594f2f3dfb98d75dba1b3b6a4a1d627e` |
| docs/gpu_training_container.md | 3143 | `67a0719e63b9ef20888fb0f64c508a01ea1a1692544c16957c71c9c51ed7a8d9` |
| docs/local_gpu_container_validation.md | 892 | `1843f9175623363f1d58804833a0d56410b3102eb9c07606d3c08dbf986bdd37` |
| docs/compute_resource_accounting.md | 5661 | `9e11c26e55eea6f6de347cdd3e60af83abd305b301943c0bf2cd6f188f18d9ad` |
| docs/asset_license_inventory.md | 5606 | `85d1899b4c49210925592193f43e094419f17cc075dce2213b9dc4ae13c54bbd` |
| docs/new_asset_inventory.md | 5131 | `b542416e55a37fb1d990c7171dd11c31a2cfa7590a3e83d1ec3eb0e578398494` |
| docs/cifar10_resnet20_full_covariance_feasibility.md | 3098 | `4ed64e4c5cf5aaba16ccb217e8bfba2587f4ef7d7bdf42c5ad2ce38a4a6e9e15` |
| docs/digits_fullnet_laplace_tiny_r2_p0p3.md | 1318 | `aef18d08ffa61da26be8dc3d761e0ff0fd5844a68cbb57a68363de80aedde0e5` |
| docs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke.md | 1154 | `a1e0dc159eec46d8da8b69836d4c5849079711c3f5026810142061c0031cb212` |
| docs/linear_connectivity_barrier_audit.md | 1751 | `b264dc04e46c82ed4072dc1e2c8c0d20e838f17c29fa1ac0ec7ca052f425e78c` |
| docs/posterior_covariance_robustness_audit.md | 3904 | `37790b1a3f600b945b20f560491d7445c339b845124c125b577f63d493327362` |
| docs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3.md | 1293 | `7305cde802d613561340a00c6be5a457fa1cbc6dab2a9350b41f3ca3723b0792` |
| docs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3.md | 1293 | `f977cee93ab8b479945c648c29bd3a0fec2167841570f872541e5dfb4da77295` |
| docs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3.md | 1292 | `1d3d0994d444a477d778bb2404c4f5df3ce96732a251cab7463c7b85d2ff93b9` |
| docs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3.md | 824 | `4de29cb678af6b604d3c5d255834e5803aa97fe04baf26a6202bd66fbac07cfe` |
| docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3.md | 1086 | `1874665908f405a1846764746e0e8551e064f90f539c563363bdc74fc607cc75` |
| docs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3.md | 1088 | `1965e84f22637256cd7d2fd868c2bd080e27531fc5018355d3ec227f5698e854` |
| docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3.md | 1194 | `a83c298ef34a6eb0fdb2bd8e62c50d1067cb071b0fd62a6ff3d7b05f33404ff6` |
| docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3.md | 1194 | `e970b9864259330f8bae509aab27dff99fb5378cc7a50826a1cad35cc911db1a` |
| docs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3.md | 1194 | `60eec9801289ee82c77cba6bae603b05ee195d1d4e21e5f0f3516d830c1dc4ef` |
| docs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3.md | 857 | `0ae7dc6358f66ce19506ba4ed22563de8805469230b78320960edd62bba33a0c` |
| docs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3.md | 587 | `f7a4cef2df0708bf95ea1408c805b3c34f97e65f112120b7b7782528aaf15546` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3.md | 3367 | `ddef34956dd52eab6f8a245e78763cc49ec4cd8d9214238d78dd6bd578eec31c` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3.md | 2722 | `974fab23451c502712161a64d62ad3738790cae5c02e5c7c462c9e37e45c6584` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3.md | 2702 | `d156f190b4a11bbb877f2882890d2f6d7ac385a62a8bc1a644bf90335f1e8274` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3.md | 2716 | `c23a31470f25da234a0c0e019b54bcc75b1f500d04673397db0f784317d8a07c` |
| docs/fake_cifar10_mode_ticket_mask_artifact_smoke.md | 4274 | `2652c88771a9d46551b85e760269554c0bd633b365443689380fefe92aafc505` |
| docs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.md | 3698 | `96ab348b821e6edff5c80718a8634c6c6bda4ba4226219fbff0f2c7bfdb8b316` |
| docs/mode_ticket_artifact_storage_budget.md | 2913 | `0ad534be0a775f5a9ac6a1fc27214d3d270ce649a480de3cb716bb7313f951e2` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3.md | 3397 | `7a1341135bd6d4f64261380eb33acb4c3337ea4504814b399ee25c8155256583` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.md | 3486 | `ff41f6eb74fdc44f5f685c890c470fe189504b7d504d621ce20d15ecde85b199` |
| docs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.md | 2021 | `02605047cff6d6498627fd2d02ac027fa8aaab0f25b6875c9042633bc17c30a2` |
| docs/resnet_channel_permutation_exhaustive_feasibility_audit.md | 2516 | `bc2c7f5e20f3a156fe4a8cad1eaa2f45632f3b87e64f55e4bdeaa5603c3f86dd` |
| docs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3.md | 1572 | `e9a55cd09442206adb386711642169a56ea98c4d6404d63858fc732f22e1ee6f` |
| docs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3.md | 1506 | `49244555b681c1c77cde204cc1165f33a791019e2c99cebd820ba62b9d70426b` |
| docs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3.md | 1516 | `725c1ad0bc3d08e6e1bb4c04ec6c9a03609f3e746f6777940fa5cfc21196fad5` |
| docs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3.md | 1523 | `49434006a530b741a3f1d5acce90f841afa7134cd889eebe9650dc258794903d` |
| docs/environment_lock.json | 527 | `9cccbbc8d32de011ea78ba5b5b8a148e0c923e26cc8feb90a08d105a8f388615` |
| docs/mode_ticket_alignment_artifact_audit.md | 4154 | `d8a28aa378d015e1464ebf46f2d38a5e5dd518a68088e9421a63401f507646d7` |
| docs/paper_claim_ledger.md | 15276 | `6fb95ff1c4e0a2d98816512f9c5bb8dd2a151d3561cb6f20a16ddd2ec20fb387` |
| docs/paper_submission_shape_audit.md | 1808 | `f9aa52226537adc7f70e60ad18f7aa60eecb11cfa52a6b53d8032c0d0576bd65` |
| docs/submission_pdf_shape_audit.md | 668 | `48b21d08053262e002901c90223b29dd4376292326827f71864c1ff955da1ea8` |
| docs/venue_submission_compliance_audit.md | 2409 | `e4c36233b7e803fa082bed354bf385aba7227f8a73a152e8b31c956dd139de26` |
| docs/reviewer_objection_matrix.md | 8918 | `6b066bc5d378820443e22ba77984802f709e2bc7c67d943122ee273bfde7d330` |
| docs/reproducibility_manifest.md | 25727 | `f51df9bc04858a2fc40536912068aa6b9fa6ec3f39be1e58ee23fff69f978bf4` |
| docs/submission_readiness_audit.md | 64525 | `bf09911c57541f29e3ed8bc0cad76df7be528f5af2c4ab86597642730ef3bff3` |
| docs/thread_goal_completion_audit.md | 38103 | `cd939445485caa2685e53679b7a9333feb178d661c7d42d6655459736caa5686` |
| runs/mode_ticket_alignment_artifact_audit.json | 18515 | `e55ddc90158cad4fe2bb8384449111496d6fe7f720b3627300f5283c5f4c708b` |
| runs/fake_cifar10_mode_ticket_mask_artifact_posthoc_audit.json | 29829 | `7a9baf91f9a82b1990d41ec35c1e93c3502e0c33a545f758f4b74f8e6674036e` |
| runs/mode_ticket_artifact_storage_budget.json | 6129 | `cc35a0c9784a8af76b818febd23da3ab962bfff2c8b341f078a4a0ab78399389` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_posthoc_audit.json | 24797 | `5294c38de9f07ca0be70d513605085c7fd29a0047f00081d3cc01ad2c7336648` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_global_channel_audit.json | 46706 | `0bc66a5e667eb6e38a8f0e2575d2b689f1ee501ac3a2d83641b6fc5684404d88` |
| runs/resnet_channel_permutation_exhaustive_feasibility_audit.json | 9264 | `81d5c927e6c15be48dbc85d222d669019c578cbe6968235b73550a09320005ce` |
| runs/paper_submission_shape_audit.json | 2508 | `9f6e8fc984abc32ebc568cb71e85d72f1879ea93af124f9be9f725ca193417dd` |
| runs/submission_pdf_shape_audit.json | 321 | `dfb09b0b5026b75327535733de666df2e8377625d6380d5b615ad36f3b35e52f` |
| runs/venue_submission_compliance_audit.json | 2358 | `f06caa6f9e04b920262f5c8f117c6d81b864e21c6627685480e43d44aec5ea2f` |
| runs/local_gpu_container_validation.json | 1851 | `51e7c6cbf666ced3b1ad2b9973bd11190d21861d76b8903bb23e2bbc221d4083` |
| runs/reviewer_objection_matrix.json | 10770 | `d3f29151314bbe7e7210025ca0991b2034da3fe5c7ff4e37a126765f91b7ace8` |
| runs/paper_stats.json | 1768945 | `982a927407be2bccabb8fa826c4586181927210b8497d964b493aaf8a9243c3b` |
| paper/main.tex | 28625 | `f8ff587d524ea6780cdabf8c9567e421cbd3468941cf1c21548d0bff9d54713b` |
| paper/refs.bib | 4809 | `23792ddea0ba5742bda7843ad76b399caf3ff680531f43fd3a2f2bb8e4a23f6c` |
| paper/main.pdf | 355521 | `e595a2de035fda189ec6588d25f748bc94d4d69c8906e45eb381dab2907171cd` |
| paper/main_submission.pdf | 266517 | `224e9c6526fa8204c4b1f33cf615c4d6d25a0badafee4a95d45c909eda747368` |
| paper/neurips_2026.sty | 13462 | `0c1ad36961fcd9198dcc2558cf2793e1df39973bde8264fd701f5e7970672757` |
| paper/neurips_checklist.tex | 6682 | `2b955470cf337a5bffb4867dca725917d97de78193bcb113837090c1993db217` |
| paper/neurips_submission.pdf | 292374 | `73e0115bb3c356ce83e12d6f014bde076d600e1b0c1b0a56d1bca9f4acb31cce` |
| paper/tables/statistical_summary.tex | 32518 | `03125756250f7b1c6f37ae63493c591b34bfe4a005099926032cbcb3eba3cb49` |
| scripts/audit_mode_ticket_alignment_artifacts.py | 21120 | `075c8aeab29ca7b9b25c20b5643220085296c32a764b26a022d097591e3d939a` |
| scripts/audit_mask_artifact_posthoc_matching.py | 35055 | `3cbde03a342226555cb151803616b095640637923f72dadbfb0a9c0f47d7e52f` |
| scripts/audit_full_data_channel_permutation_matching.py | 25065 | `c5a8e9e1b2ee8652cefa5f008d1791742060c492d4fb43a4a010910dcb81337e` |
| scripts/audit_exhaustive_channel_permutation_feasibility.py | 16690 | `9b1d6ecb539b8b5d3f284a6b65e5d8a5bd453b98c083e8fc8a7f8c93881ddd48` |
| scripts/audit_mode_ticket_artifact_storage_budget.py | 11903 | `9715e298f946c568ec604d57f8bc8452ac6cbe5f52c6162e83b849b26596e7c0` |
| scripts/run_digits_fullnet_laplace_probe.py | 12631 | `9a0343d072526466943bec76db4f649cf803bbab776ec4dd0106122bb81d2f92` |
| scripts/summarize_fullnet_laplace_probe.py | 8794 | `58419e9bb6358248d56a63ec197f381f6179441ec97b490efb494d8093069cde` |
| scripts/audit_linear_connectivity_barriers.py | 13607 | `bd7fa09e0777b18c1e3bc239323d74dd106f0641516a62f839387364d3e5b204` |
| scripts/audit_posterior_covariance_robustness.py | 22604 | `58a8d434fa3de340f6c884c4ff613dd6b5d203954537fc57db201787ad287e1e` |
| src/lottery/full_laplace.py | 7391 | `66228aea0415e59ba3d6565d70df79c57bb4c1926eadd3906187b157dc968bcc` |
| scripts/build_paper_claim_ledger.py | 63527 | `fb246e2d6c259dcea206b25fc947bb6d53aa51cf014de932a669073454f3d7fe` |
| scripts/audit_paper_submission_shape.py | 12089 | `05cbffaee4506e6fb5fd556bf032a3d3c3ddee0c339e8f9a2057e704f2bb9714` |
| scripts/audit_submission_pdf_shape.py | 3983 | `a8a815b2f41a9ce0478865a0a9f6b5d9916c1c176f0e696f57a3fc76c7648b25` |
| scripts/audit_venue_submission_compliance.py | 25962 | `aef56c440f5e4717a21d89571f29fb185207d017cac19542456432c9b70bd983` |
| scripts/build_reviewer_objection_matrix.py | 27266 | `ec9d2717c031402280c39ff876db72dc8894f923fdfaef5f84685d335ab692ab` |
| scripts/audit_external_validation_readiness.py | 24620 | `e4f782d37ab670d16acc9bb610b0cf52771aba75321dfce4111329a11c50abd5` |
| scripts/build_external_validation_receipt_template.py | 12065 | `d20183b1b4ea2ca7defbf85363fcff6dd0803830c55562762e218641fb16212d` |
| scripts/update_external_validation_receipts.py | 10049 | `b882a7c845989650f45046eed4c285eb7eff5391390759498ea3f380e4d81b65` |
| scripts/build_external_validation_runbook.py | 12864 | `3a25c7231b4578c529542f9598dbd4028efa9e2d83bb0d66f0f22591899a51e5` |
| scripts/build_submission_handoff.py | 12733 | `122259a736fba07b63646009a6492ef244a25db6e6390bee56a69b6a630b5300` |
| scripts/stage_public_repository_snapshot.py | 15081 | `854622d399515d199b9f384acaa5222f75a2a726a6dfda5dd5add94d1aa97a89` |
| scripts/smoke_public_repository_snapshot.py | 7076 | `3f19ea04e3b9b38dda07ce28034306e18908adf10ced4c62d30a2e3c9423b974` |
| scripts/verify_source_repository_snapshot.py | 6836 | `053e1c8534a943108986fb6b29e643265440d8a2ef0c08b6b3ca77f72204605a` |
| scripts/audit_release_anonymization.py | 8042 | `42b06f7cd849c903df3ff6b2baf1c9a23c936c2b3f5b348feb417e923ea53b72` |
| scripts/build_public_release_archive.py | 11544 | `e285b16871910a68ec77fc34e54310677e8a5a6358805ba4608f3932c5584e0f` |
| scripts/smoke_public_release_archive.py | 9254 | `c480f1a9c7972c5005719b585bf5e4ae99c79c5c0d1f8b92da5db04cb611ccaf` |
| scripts/check_gpu_training_environment.py | 6063 | `9c507fc918e7c0038eb9af292aab42b25a2f627d1b2bc75612537a65610cb35c` |
| scripts/run_gpu_container_env_check.py | 2418 | `877136f81b1b6904b903f8de748a5bdca787cb38285831a4cb28b2865209d5a3` |
| scripts/build_local_gpu_container_validation.py | 6908 | `c9f6b2be14bb088dfaa620b65381c58c0cb3fd7f5f6668ffa7b66adbc2e7613b` |
| runs/cifar10_resnet20_full_covariance_feasibility.json | 6084 | `ef572b0733d9e6299bd8ea12146921b710e71c6c7f49e5183652708b40a3a1ac` |
| runs/digits_fullnet_laplace_tiny_r2_p0p3_summary.csv | 5704 | `25c9daa9df2b9446e47bffb38dd32ba9b7e51f073e6c5367191a03d721da6af6` |
| runs/fake_cifar10_resnet20_w1_fullnet_laplace_smoke_summary.csv | 3679 | `6bdb5373f5fe0163c0647b38e6c00b136f8eb4c24497915c833339a7dd07f045` |
| runs/linear_connectivity_barrier_audit.csv | 9146 | `7d89083538c34e68d53eb54136c302992f7fa3f08474cd869a5f62fe02fa45d3` |
| runs/linear_connectivity_barrier_audit.json | 22718 | `07cde4f12d0b89412bf848106bd38aef8274ff009674f907dd905ff42fa8ab5a` |
| runs/posterior_covariance_robustness_audit.csv | 4264 | `8722067be7b082994abd8e63c3e6c5366d596972652a8350cfc944ceb77aa2a9` |
| runs/posterior_covariance_robustness_audit.json | 10394 | `60ac7cb413ca8d8cc593e0f60e3dc97ea2421af47dcac54682f74bdcf69a1734` |
| runs/cifar10_resnet20_long30_rewind1_lowrank_laplace_movement_selected_r5_p0p3_summary.csv | 3504 | `7ee0c848ca08059fe0e39e8b34fe6b19a3a3ffea9c9708eaf9df76e3cd1518f8` |
| runs/cifar10_resnet20_long30_rewind1_lowrank32_laplace_movement_selected_r5_p0p3_summary.csv | 3586 | `ae4be8214e2731f58a3c297d16e0431a09579dea8b1044133a2ca0e04330eb2f` |
| runs/cifar10_resnet20_long30_rewind1_lowrank64_laplace_movement_selected_r5_p0p3_summary.csv | 3560 | `e99758687b18659981c66eb35a216db48cd68e43e35c62fd578cba4840c5f587` |
| runs/cifar10_resnet20_long30_rewind1_lowrank128_laplace_movement_selected_r5_p0p3_summary.csv | 1643 | `3c2e0e5ed4e347784d26194c44dd37d990fc28329ea06dc7a544d4d508afc8b6` |
| runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_selected_r5_p0p3_summary.csv | 2397 | `2a1e21c17b55e72b41589a47026f0f893eeb847f0c203e1bb412ce455cac6659` |
| runs/cifar10_resnet20_long30_rewind1_blockdiag_laplace_max10k_selected_r5_p0p3_summary.csv | 2384 | `68fe14d9524410a63a1c64b5fc42a6d5b8767fbce572ba0800c4177c6135b23c` |
| runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max10k_selected_r5_p0p3_summary.csv | 2392 | `fdcd9ba3865d14a5fca8d65be1d9d32ee24374a21fd2348ccf17b7e65a6a38cd` |
| runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max20k_selected_r5_p0p3_summary.csv | 2403 | `66696179a81b9ddeb06ed10738e868b1ff54a89bf55a7de00c0ea0a93fb25db9` |
| runs/cifar10_resnet20_long30_rewind1_jointdiag_laplace_max40k_stream_selected_r5_p0p3_summary.csv | 2417 | `4ee97f479d458cc83b59758045fa86b68d4a4b58aa936cc3a3f6cf7edfd4bc91` |
| runs/cifar10_resnet20_long30_rewind1_hessian32_subspace_hmc_selected_r5_p0p3_summary.csv | 2172 | `cd8b0ccbf311533cd4f497ea61ccd35b665cc26c6a01e7f55baf6c55b5be8742` |
| runs/cifar10_resnet20_long30_rewind1_hard_concrete_selected_r5_p0p3_summary.csv | 2023 | `fa1e0f2e4a3a3dd531abb2f485541c81f8dcea193051d7e427499612bca3162b` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_weight_aligned_r5_p0p3_summary.csv | 2844 | `b0fe9556e52aef131a9dbb9a69c096a3022aeaf468c50654e1bad6901c14b8bc` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_csgld_independent_multichain_r5_p0p3_summary.csv | 1727 | `e1225b193eed60c474e768a560fbbb721f76eb82afbe101d4b18f0344dec7103` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_lowrank128_laplace_r5_p0p3_summary.csv | 1714 | `a6d1420d879f2d79cf7bf5b72a1333a15a58651ef912c89f3b8a4d2b86b246fa` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_jointdiag_laplace_max40k_stream_r5_p0p3_summary.csv | 1748 | `de7cf26011b8f73b4b8f94f2eed2f8344d573c1fc5704fb3b24d7aa415edb776` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3_summary.csv | 2947 | `1761472b8ad898d2006c5f0effa4dac2c3e245104e680c3723946e7b22cd3d78` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/metrics.json | 80728 | `462d02873f9da2d697ceda64253f1f6e4234d4ebcc5b94448982c183a27750af` |
| runs/cifar10_resnet20_long30_rewind1_mode_ticket_distribution_activation_aligned_saved_artifacts_r5_p0p3/20260506_230706/mask_artifacts.npz | 118946673 | `0918c8795ccc01f5896ac1b0ba6c181a415fef31a805990bbfd6fb81061a7843` |
| runs/fake_cifar10_mode_ticket_mask_artifact_smoke_summary.csv | 2724 | `1b242a8d392bdcdf1b0b3e4520b916385251ad5f180824ecc5fd1ecb9af1b507` |
| runs/cifar10_resnet20_long30_rewind1_residual_imp_process_stratified_exclusion_r5_p0p3_summary.csv | 3935 | `725da163f9a3048aa03982c6431294b90bee762ceda614ade5fe53ea54b43711` |
| runs/cifar10_resnet20_long30_rewind1_residual_imp_process_projection_r5_p0p3_summary.csv | 3873 | `4b29c114215fb1e258dcb284de9e3de39031bf61202d28f38a2c161642e42dde` |
| runs/cifar10_resnet20_long30_rewind1_residual_imp_process_posterior_projection_r5_p0p3_summary.csv | 3896 | `339e324e6ed8231cf6ee4f6fba3b3b809bdee9863117a89019a5db032e27bb5f` |
| runs/cifar10_resnet20_long30_rewind1_residual_imp_process_learned_subspace_r5_p0p3_summary.csv | 3930 | `7eed5700c5cd83e43776d062a7be4f43bf4dfe1a040ef4c60602e23dc23b0824` |

## Verification

```bash
make check
make paper-check
make paper-neurips-check
```

This file is generated by `scripts/build_release_manifest.py`.
