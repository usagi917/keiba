from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def wilson_interval(successes: np.ndarray, n: int, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    p = successes / max(n, 1)
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4.0 * (n ** 2))) / denom
    lower = np.clip(center - margin, 0.0, 1.0)
    upper = np.clip(center + margin, 0.0, 1.0)
    return lower, upper


def monte_carlo_rank_distribution(
    horse_names: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    names = np.asarray(horse_names)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    n_horses = len(names)
    rng = np.random.default_rng(seed)

    total_trials = 0
    win_counts = np.zeros(n_horses, dtype=np.int64)
    top2_counts = np.zeros(n_horses, dtype=np.int64)
    top3_counts = np.zeros(n_horses, dtype=np.int64)
    rank_sums = np.zeros(n_horses, dtype=np.int64)
    rank_counts = np.zeros((n_horses, n_horses), dtype=np.int64)

    prev_top3_prob: np.ndarray | None = None
    diagnostics: List[Dict[str, float]] = []

    for stage in stages:
        additional = int(stage - total_trials)
        if additional <= 0:
            continue

        while additional > 0:
            batch = min(block_size, additional)
            scores = rng.normal(loc=mu_arr.reshape(1, -1), scale=sigma_arr.reshape(1, -1), size=(batch, n_horses))
            order = np.argsort(-scores, axis=1)
            ranks = np.empty_like(order)
            ranks[np.arange(batch)[:, None], order] = np.arange(1, n_horses + 1)

            win_counts += (ranks == 1).sum(axis=0)
            top2_counts += (ranks <= 2).sum(axis=0)
            top3_counts += (ranks <= 3).sum(axis=0)
            rank_sums += ranks.sum(axis=0)
            for horse_idx in range(n_horses):
                rank_counts[horse_idx] += np.bincount(ranks[:, horse_idx] - 1, minlength=n_horses)

            total_trials += batch
            additional -= batch

        current_top3_prob = top3_counts / total_trials
        max_delta = float(np.max(np.abs(current_top3_prob - prev_top3_prob))) if prev_top3_prob is not None else np.nan
        diagnostics.append(
            {
                "trials": float(total_trials),
                "max_top3_delta": max_delta,
            }
        )
        if prev_top3_prob is not None and max_delta <= threshold:
            break
        prev_top3_prob = current_top3_prob.copy()

    win_prob = win_counts / total_trials
    top2_prob = top2_counts / total_trials
    top3_prob = top3_counts / total_trials
    mean_rank = rank_sums / total_trials
    ci_low, ci_high = wilson_interval(top3_counts, total_trials)

    result = pd.DataFrame(
        {
            "horse_name": names,
            "mu": mu_arr,
            "sigma": sigma_arr,
            "win_prob": win_prob,
            "top2_prob": top2_prob,
            "top3_prob": top3_prob,
            "mean_rank": mean_rank,
            "top3_ci_low": ci_low,
            "top3_ci_high": ci_high,
            "top3_ci_width": ci_high - ci_low,
            "trials_used": total_trials,
        }
    )

    for rank_no in range(1, n_horses + 1):
        result[f"rank_{rank_no}_prob"] = rank_counts[:, rank_no - 1] / total_trials

    diagnostics_df = pd.DataFrame(diagnostics)
    return result.sort_values("top3_prob", ascending=False).reset_index(drop=True), diagnostics_df


def normalize_scores_by_group(group_ids: Sequence[object], scores: Sequence[float]) -> np.ndarray:
    group_arr = pd.Series(group_ids, dtype="string")
    score_arr = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    out = np.full(len(score_arr), 0.0, dtype=float)

    for _, idx in group_arr.groupby(group_arr, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        group_scores = score_arr[idx_arr]
        valid = np.isfinite(group_scores)
        if valid.sum() == 0:
            out[idx_arr] = 0.0
            continue
        valid_scores = group_scores[valid]
        center = float(np.mean(valid_scores))
        scale = float(np.std(valid_scores))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = np.zeros_like(group_scores, dtype=float)
        normalized[valid] = (valid_scores - center) / scale
        out[idx_arr] = normalized

    return out


def _simulate_rank_counts(
    strengths: np.ndarray,
    temperature: float,
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int,
) -> Tuple[Dict[str, np.ndarray | int], pd.DataFrame]:
    strength_arr = np.asarray(strengths, dtype=float)
    temp = max(float(temperature), 1e-6)
    n_horses = len(strength_arr)
    rng = np.random.default_rng(seed)

    total_trials = 0
    win_counts = np.zeros(n_horses, dtype=np.int64)
    top2_counts = np.zeros(n_horses, dtype=np.int64)
    top3_counts = np.zeros(n_horses, dtype=np.int64)
    rank_sums = np.zeros(n_horses, dtype=np.int64)
    rank_counts = np.zeros((n_horses, n_horses), dtype=np.int64)

    prev_top3_prob: np.ndarray | None = None
    diagnostics: List[Dict[str, float]] = []

    for stage in stages:
        additional = int(stage - total_trials)
        if additional <= 0:
            continue

        while additional > 0:
            batch = min(block_size, additional)
            sampled_scores = rng.gumbel(
                loc=strength_arr.reshape(1, -1),
                scale=temp,
                size=(batch, n_horses),
            )
            order = np.argsort(-sampled_scores, axis=1)
            ranks = np.empty_like(order)
            ranks[np.arange(batch)[:, None], order] = np.arange(1, n_horses + 1)

            win_counts += (ranks == 1).sum(axis=0)
            top2_counts += (ranks <= 2).sum(axis=0)
            top3_counts += (ranks <= 3).sum(axis=0)
            rank_sums += ranks.sum(axis=0)
            for horse_idx in range(n_horses):
                rank_counts[horse_idx] += np.bincount(ranks[:, horse_idx] - 1, minlength=n_horses)

            total_trials += batch
            additional -= batch

        current_top3_prob = top3_counts / total_trials
        max_delta = float(np.max(np.abs(current_top3_prob - prev_top3_prob))) if prev_top3_prob is not None else np.nan
        diagnostics.append(
            {
                "trials": float(total_trials),
                "max_top3_delta": max_delta,
            }
        )
        if prev_top3_prob is not None and max_delta <= threshold:
            break
        prev_top3_prob = current_top3_prob.copy()

    return {
        "win_counts": win_counts,
        "top2_counts": top2_counts,
        "top3_counts": top3_counts,
        "rank_sums": rank_sums,
        "rank_counts": rank_counts,
        "total_trials": total_trials,
    }, pd.DataFrame(diagnostics)


def _simulate_gaussian_rank_counts(
    mu: np.ndarray,
    sigma: np.ndarray,
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int,
) -> Tuple[Dict[str, np.ndarray | int], pd.DataFrame]:
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    n_horses = len(mu_arr)
    rng = np.random.default_rng(seed)

    total_trials = 0
    win_counts = np.zeros(n_horses, dtype=np.int64)
    top2_counts = np.zeros(n_horses, dtype=np.int64)
    top3_counts = np.zeros(n_horses, dtype=np.int64)
    rank_sums = np.zeros(n_horses, dtype=np.int64)
    rank_counts = np.zeros((n_horses, n_horses), dtype=np.int64)

    prev_top3_prob: np.ndarray | None = None
    diagnostics: List[Dict[str, float]] = []

    for stage in stages:
        additional = int(stage - total_trials)
        if additional <= 0:
            continue

        while additional > 0:
            batch = min(block_size, additional)
            sampled_scores = rng.normal(
                loc=mu_arr.reshape(1, -1),
                scale=sigma_arr.reshape(1, -1),
                size=(batch, n_horses),
            )
            order = np.argsort(-sampled_scores, axis=1)
            ranks = np.empty_like(order)
            ranks[np.arange(batch)[:, None], order] = np.arange(1, n_horses + 1)

            win_counts += (ranks == 1).sum(axis=0)
            top2_counts += (ranks <= 2).sum(axis=0)
            top3_counts += (ranks <= 3).sum(axis=0)
            rank_sums += ranks.sum(axis=0)
            for horse_idx in range(n_horses):
                rank_counts[horse_idx] += np.bincount(ranks[:, horse_idx] - 1, minlength=n_horses)

            total_trials += batch
            additional -= batch

        current_top3_prob = top3_counts / total_trials
        max_delta = float(np.max(np.abs(current_top3_prob - prev_top3_prob))) if prev_top3_prob is not None else np.nan
        diagnostics.append(
            {
                "trials": float(total_trials),
                "max_top3_delta": max_delta,
            }
        )
        if prev_top3_prob is not None and max_delta <= threshold:
            break
        prev_top3_prob = current_top3_prob.copy()

    return {
        "win_counts": win_counts,
        "top2_counts": top2_counts,
        "top3_counts": top3_counts,
        "rank_sums": rank_sums,
        "rank_counts": rank_counts,
        "total_trials": total_trials,
    }, pd.DataFrame(diagnostics)


def plackett_luce_rank_distribution(
    horse_names: Sequence[str],
    strengths: np.ndarray,
    temperature: float,
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    names = np.asarray(horse_names)
    metrics, diagnostics_df = _simulate_rank_counts(
        strengths=np.asarray(strengths, dtype=float),
        temperature=temperature,
        stages=stages,
        threshold=threshold,
        seed=seed,
        block_size=block_size,
    )

    total_trials = int(metrics["total_trials"])
    win_counts = np.asarray(metrics["win_counts"], dtype=np.int64)
    top2_counts = np.asarray(metrics["top2_counts"], dtype=np.int64)
    top3_counts = np.asarray(metrics["top3_counts"], dtype=np.int64)
    rank_sums = np.asarray(metrics["rank_sums"], dtype=np.int64)
    rank_counts = np.asarray(metrics["rank_counts"], dtype=np.int64)
    strength_arr = np.asarray(strengths, dtype=float)

    win_prob = win_counts / total_trials
    top2_prob = top2_counts / total_trials
    top3_prob = top3_counts / total_trials
    mean_rank = rank_sums / total_trials
    ci_low, ci_high = wilson_interval(top3_counts, total_trials)

    result = pd.DataFrame(
        {
            "horse_name": names,
            "strength": strength_arr,
            "temperature": float(max(temperature, 1e-6)),
            "win_prob": win_prob,
            "top2_prob": top2_prob,
            "top3_prob": top3_prob,
            "mean_rank": mean_rank,
            "top3_ci_low": ci_low,
            "top3_ci_high": ci_high,
            "top3_ci_width": ci_high - ci_low,
            "trials_used": total_trials,
        }
    )

    for rank_no in range(1, len(names) + 1):
        result[f"rank_{rank_no}_prob"] = rank_counts[:, rank_no - 1] / total_trials

    return result.sort_values("top3_prob", ascending=False).reset_index(drop=True), diagnostics_df


def gaussian_group_rank_distribution(
    entity_ids: Sequence[object],
    group_ids: Sequence[object],
    mu: Sequence[float],
    sigma: Sequence[float],
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entity_arr = np.asarray(entity_ids)
    group_series = pd.Series(group_ids, dtype="string")
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    result_frames: List[pd.DataFrame] = []
    diag_frames: List[pd.DataFrame] = []

    for group_no, (group_id, idx) in enumerate(group_series.groupby(group_series, sort=False).groups.items()):
        idx_arr = np.asarray(list(idx), dtype=int)
        group_result, group_diag = monte_carlo_rank_distribution(
            horse_names=entity_arr[idx_arr],
            mu=mu_arr[idx_arr],
            sigma=sigma_arr[idx_arr],
            stages=stages,
            threshold=threshold,
            seed=seed + group_no,
            block_size=block_size,
        )
        group_result["group_id"] = str(group_id)
        group_diag["group_id"] = str(group_id)
        result_frames.append(group_result.rename(columns={"horse_name": "entity_id"}))
        diag_frames.append(group_diag)

    if not result_frames:
        return pd.DataFrame(), pd.DataFrame()

    result_df = pd.concat(result_frames, ignore_index=True)
    result_df = result_df.sort_values("entity_id").reset_index(drop=True)
    diagnostics_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    return result_df, diagnostics_df


def plackett_luce_group_rank_distribution(
    entity_ids: Sequence[object],
    group_ids: Sequence[object],
    strengths: Sequence[float],
    temperature: float,
    stages: Sequence[int],
    threshold: float,
    seed: int,
    block_size: int = 50_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entity_arr = np.asarray(entity_ids)
    group_series = pd.Series(group_ids, dtype="string")
    strength_arr = np.asarray(strengths, dtype=float)

    result_frames: List[pd.DataFrame] = []
    diag_frames: List[pd.DataFrame] = []

    for group_no, (group_id, idx) in enumerate(group_series.groupby(group_series, sort=False).groups.items()):
        idx_arr = np.asarray(list(idx), dtype=int)
        group_result, group_diag = plackett_luce_rank_distribution(
            horse_names=entity_arr[idx_arr],
            strengths=strength_arr[idx_arr],
            temperature=temperature,
            stages=stages,
            threshold=threshold,
            seed=seed + group_no,
            block_size=block_size,
        )
        group_result["group_id"] = str(group_id)
        group_diag["group_id"] = str(group_id)
        result_frames.append(group_result.rename(columns={"horse_name": "entity_id"}))
        diag_frames.append(group_diag)

    if not result_frames:
        return pd.DataFrame(), pd.DataFrame()

    result_df = pd.concat(result_frames, ignore_index=True)
    result_df = result_df.sort_values("entity_id").reset_index(drop=True)
    diagnostics_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    return result_df, diagnostics_df


def plackett_luce_group_topk_probabilities(
    group_ids: Sequence[object],
    strengths: Sequence[float],
    topk: int,
    temperature: float,
    n_trials: int,
    seed: int,
    block_size: int = 20_000,
) -> np.ndarray:
    group_series = pd.Series(group_ids, dtype="string")
    strength_arr = np.asarray(strengths, dtype=float)
    out = np.full(len(strength_arr), np.nan, dtype=float)

    stages = [int(max(n_trials, 1))]
    for group_no, (_, idx) in enumerate(group_series.groupby(group_series, sort=False).groups.items()):
        idx_arr = np.asarray(list(idx), dtype=int)
        metrics, _ = _simulate_rank_counts(
            strengths=strength_arr[idx_arr],
            temperature=temperature,
            stages=stages,
            threshold=0.0,
            seed=seed + group_no,
            block_size=block_size,
        )
        total_trials = int(metrics["total_trials"])
        if topk <= 1:
            counts = np.asarray(metrics["win_counts"], dtype=np.int64)
        elif topk == 2:
            counts = np.asarray(metrics["top2_counts"], dtype=np.int64)
        else:
            counts = np.asarray(metrics["top3_counts"], dtype=np.int64)
        out[idx_arr] = counts / total_trials

    return out


def gaussian_group_topk_probabilities(
    group_ids: Sequence[object],
    mu: Sequence[float],
    sigma: Sequence[float],
    topk: int,
    n_trials: int,
    seed: int,
    block_size: int = 20_000,
) -> np.ndarray:
    group_series = pd.Series(group_ids, dtype="string")
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    out = np.full(len(mu_arr), np.nan, dtype=float)

    stages = [int(max(n_trials, 1))]
    for group_no, (_, idx) in enumerate(group_series.groupby(group_series, sort=False).groups.items()):
        idx_arr = np.asarray(list(idx), dtype=int)
        metrics, _ = _simulate_gaussian_rank_counts(
            mu=mu_arr[idx_arr],
            sigma=sigma_arr[idx_arr],
            stages=stages,
            threshold=0.0,
            seed=seed + group_no,
            block_size=block_size,
        )
        total_trials = int(metrics["total_trials"])
        rank_counts = np.asarray(metrics["rank_counts"], dtype=np.int64)
        effective_topk = min(max(int(topk), 1), rank_counts.shape[1])
        counts = rank_counts[:, :effective_topk].sum(axis=1)
        out[idx_arr] = counts / total_trials

    return out
