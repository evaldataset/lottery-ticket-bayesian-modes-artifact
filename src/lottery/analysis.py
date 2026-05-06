from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

from lottery.masks import Mask, hamming_similarity, random_mask_like, support_jaccard


def flatten_state(
    state: Mapping[str, torch.Tensor],
    names: list[str],
) -> np.ndarray:
    parts = [state[name].detach().cpu().reshape(-1).numpy() for name in names]
    return np.concatenate(parts)


def cluster_states(
    states: list[Mapping[str, torch.Tensor]],
    names: list[str],
    pca_dim: int = 20,
) -> dict[str, float]:
    if len(states) < 3:
        return {"num_clusters": float(len(states)), "noise_fraction": 0.0}

    matrix = np.stack([flatten_state(state, names) for state in states], axis=0)
    return cluster_matrix(matrix, pca_dim=pca_dim)


def cluster_matrix(matrix: np.ndarray, pca_dim: int = 20) -> dict[str, float]:
    if matrix.shape[0] < 3:
        return {"num_clusters": float(matrix.shape[0]), "noise_fraction": 0.0}
    n_components = min(pca_dim, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    bandwidth = estimate_bandwidth(reduced, quantile=0.3, n_samples=matrix.shape[0])
    if bandwidth <= 1e-12:
        return {"num_clusters": 1.0, "noise_fraction": 0.0}
    labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(reduced)
    return {
        "num_clusters": float(len(set(labels.tolist()))),
        "largest_cluster_fraction": float(np.bincount(labels).max() / labels.shape[0]),
    }


def overlap_rows(
    posterior_masks: list[Mask],
    imp_mask: Mask,
    sparsity: float,
    random_trials: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx, mask in enumerate(posterior_masks):
        rows.append(
            {
                "source": "posterior",
                "index": float(idx),
                "jaccard": support_jaccard(mask, imp_mask),
                "hamming": hamming_similarity(mask, imp_mask),
            }
        )
    for idx in range(random_trials):
        mask = random_mask_like(imp_mask, sparsity=sparsity, seed=seed + idx)
        rows.append(
            {
                "source": "random",
                "index": float(idx),
                "jaccard": support_jaccard(mask, imp_mask),
                "hamming": hamming_similarity(mask, imp_mask),
            }
        )
    return rows


def summarize_overlaps(rows: list[dict[str, float | str]]) -> dict[str, float]:
    posterior = [float(row["jaccard"]) for row in rows if row["source"] == "posterior"]
    random = [float(row["jaccard"]) for row in rows if row["source"] == "random"]
    posterior_arr = np.asarray(posterior, dtype=np.float64)
    random_arr = np.asarray(random, dtype=np.float64)
    posterior_var = posterior_arr.var(ddof=1) if len(posterior_arr) > 1 else 0.0
    random_var = random_arr.var(ddof=1) if len(random_arr) > 1 else 0.0
    pooled_std = np.sqrt(0.5 * (posterior_var + random_var))
    effect = (posterior_arr.mean() - random_arr.mean()) / pooled_std if pooled_std > 0 else 0.0
    rng = np.random.default_rng(0)
    boot_diffs = []
    for _ in range(2000):
        posterior_sample = rng.choice(posterior_arr, size=len(posterior_arr), replace=True)
        random_sample = rng.choice(random_arr, size=len(random_arr), replace=True)
        boot_diffs.append(float(posterior_sample.mean() - random_sample.mean()))
    ci_low, ci_high = np.quantile(np.asarray(boot_diffs), [0.025, 0.975])
    u_test = mannwhitneyu(posterior_arr, random_arr, alternative="greater")
    pairwise_win_rate = float((posterior_arr[:, None] > random_arr[None, :]).mean())
    return {
        "posterior_jaccard_mean": float(posterior_arr.mean()),
        "posterior_jaccard_std": float(posterior_arr.std(ddof=1)) if len(posterior_arr) > 1 else 0.0,
        "random_jaccard_mean": float(random_arr.mean()),
        "random_jaccard_std": float(random_arr.std(ddof=1)) if len(random_arr) > 1 else 0.0,
        "posterior_minus_random_jaccard": float(posterior_arr.mean() - random_arr.mean()),
        "posterior_minus_random_jaccard_ci95_low": float(ci_low),
        "posterior_minus_random_jaccard_ci95_high": float(ci_high),
        "jaccard_cohens_d": float(effect),
        "posterior_random_pairwise_win_rate": pairwise_win_rate,
        "mannwhitney_greater_pvalue": float(u_test.pvalue),
    }
