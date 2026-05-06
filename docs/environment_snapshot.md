# Environment Snapshot

Date: 2026-05-06

This is the local environment used for the current generated statistics and
paper build. `docs/environment_lock.json` and `requirements-lock.txt` pin the
project-critical runtime values that `scripts/check_environment_lock.py`
verifies.

## Runtime

| Component | Version / value |
| --- | --- |
| Python | 3.12.3 |
| Platform | Linux-6.17.0-20-generic-x86_64-with-glibc2.39 |
| Kernel | Linux 6.17.0-20-generic #20~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Mar 19 01:28:37 UTC 2 x86_64 |
| GPU | NVIDIA GeForce RTX 5090 |
| CUDA visible to PyTorch | 13.0 |
| `torch.cuda.is_available()` | True |
| pdfTeX | 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian) |
| BibTeX | 0.99d (TeX Live 2023/Debian) |

## Python Packages

| Package | Version |
| --- | --- |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| scikit-learn | 1.4.1.post1 |
| matplotlib | 3.6.3 |

## Standard Checks

```bash
make env-check
make check
make paper-check
```

`make env-check` checks Python, package, CUDA, and TeX versions against
`docs/environment_lock.json`. `make check` runs that lock check, Python syntax
checks, regenerates `runs/paper_stats.json`, and verifies the core evidence rows with
`scripts/verify_research_artifacts.py`.
