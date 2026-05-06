# Asset License Inventory

Date: 2026-05-07

This inventory records the third-party datasets, style files, and software
assets used by the current paper and local artifact package. Raw benchmark
datasets may exist in a local `data/` cache, but `data/` is excluded from
`.dockerignore`, `.gitignore`, and the public release manifest. The release
package redistributes generated run summaries and selected generated mask
artifacts, not raw benchmark images.

Raw benchmark datasets are treated as external assets downloaded through
documented loaders, not as files distributed by this release package.
The phrase "raw benchmark datasets" in this inventory refers to MNIST,
Fashion-MNIST, CIFAR-10, and CIFAR-100 files downloaded into local caches.

## Project License

The project-authored source code, scripts, documentation, generated summaries,
and release-package metadata are distributed under the top-level MIT License in
`LICENSE`. The anonymous-review copy uses the copyright holder
"Anonymous Authors"; a deanonymized public release can replace that holder
string after review without changing the selected license.

This project license does not relicense third-party datasets, the NeurIPS style
file, or software dependencies. Those third-party assets remain governed by
their own license or terms noted below.

## Dataset Assets

| Asset | Use in this project | Source and citation | License or terms noted for release |
| --- | --- | --- | --- |
| scikit-learn digits | Small CPU checks and dense full-network probes via `sklearn.datasets.load_digits()` | scikit-learn package and documentation; cite scikit-learn when used in the paper. | Covered through scikit-learn package terms. scikit-learn documents BSD-3-Clause source licensing for project files. |
| MNIST | Gate-1 and small image-classification controls via `torchvision.datasets.MNIST` | Original MNIST page by LeCun, Cortes, and Burges; cite LeCun et al. where relevant. | The original source page does not expose an SPDX-style dataset license in the checked page. Raw MNIST files are not redistributed by the release package. |
| Fashion-MNIST | Gate-1 and small image-classification controls via `torchvision.datasets.FashionMNIST` | Official Zalando Research GitHub repository; cite Xiao, Rasul, and Vollgraf. | The official repository lists the MIT license. Raw Fashion-MNIST files are not redistributed by the release package. |
| CIFAR-10 | Main CIFAR-10 ResNet-20 training, pruning, posterior, calibration, alignment, and residual-process evidence | Original Toronto CIFAR page and Krizhevsky technical report; UCI dataset page for metadata. | The checked UCI page points users to the linked original dataset page for licensing information, and the checked Toronto page gives citation/download instructions but no explicit SPDX-style dataset license. Raw CIFAR-10 files are not redistributed by the release package. |
| CIFAR-100 | OOD evaluation for CIFAR-10 calibration rows via `torchvision.datasets.CIFAR100` | Original Toronto CIFAR page and Krizhevsky technical report. | The checked original Toronto page describes CIFAR-100 downloads and citation but does not show an explicit dataset license. Raw CIFAR-100 files are not redistributed by the release package. |

Sources checked:

- MNIST source page: https://yann.lecun.org/exdb/mnist/
- Fashion-MNIST official repository and license: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10/CIFAR-100 original source page: https://www.cs.toronto.edu/~kriz/cifar.html
- UCI CIFAR-10 metadata: https://archive.ics.uci.edu/dataset/691/cifar+10
- NeurIPS 2026 Main Track Handbook: https://nips.cc/Conferences/2026/MainTrackHandbook

## Paper And Style Assets

| Asset | Use in this project | License or terms noted for release |
| --- | --- | --- |
| `paper/neurips_2026.sty` | Official NeurIPS 2026 submission style, copied from the official 2026 formatting instructions package. | Conference formatting asset from NeurIPS. Keep the file with its original header and do not claim it as project-created code. |
| `paper/neurips_checklist.tex` | Local checklist answers for the NeurIPS-bound submission build. | Project-authored checklist content that depends on NeurIPS checklist macros in the official style. |
| Bibliography entries in `paper/refs.bib` | Citations for datasets, methods, and baselines. | Citation metadata, not redistributed third-party data. |

## Software Dependencies

| Package | Locked version | License metadata used for release notes |
| --- | ---: | --- |
| `numpy` | 1.26.4 | Local package metadata contains BSD-3-Clause-style terms; NumPy states it is released under the modified BSD license. |
| `scipy` | 1.11.4 | Local package metadata contains BSD-3-Clause-style terms; SciPy documentation identifies the modified 3-clause BSD license. |
| `scikit-learn` | 1.4.1.post1 | Local package metadata reports `new BSD`; scikit-learn standardizes on `SPDX-License-Identifier: BSD-3-Clause`. |
| `matplotlib` | 3.6.3 | Local package metadata reports PSF; Matplotlib documentation describes a PSF-based, BSD-compatible license. |
| `torch` | 2.11.0 | Local package metadata reports BSD-3-Clause; PyTorch's repository documents BSD-style licensing in its license file. |
| `torchvision` | 0.26.0 | Local package metadata reports BSD. |

## Release Boundary

The release package includes the top-level `LICENSE` file for project-authored
assets and excludes raw benchmark datasets. Public-release upload remains a
separate external step, but the local anonymous-review package now has an
explicit project license and third-party asset boundary.
