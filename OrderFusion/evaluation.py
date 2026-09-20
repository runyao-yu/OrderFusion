"""Forecast generation, metrics, persistence, and DM comparison utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from model import predict_scaled, split_tensors
from preprocessing import DATE_COLUMN


DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_REPORT_METRICS = ("AQL", "AQCE", "AQCR", "MAE", "RMSE", "R2")
SIDE_NAMES = ("Buy", "Sell")


def evaluate_orderfusion_plus(
    model,
    prepared,
    *,
    split_name: str = "test",
    batch_size: int = 4096,
    metrics: Sequence[str] = DEFAULT_REPORT_METRICS,
    output_dir: str | Path | None = None,
    include_sides: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Predict, inverse-scale, evaluate, and optionally save one data split."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    split = getattr(prepared, split_name)
    device = next(model.parameters()).device
    values = split_tensors(split, device)
    output = predict_scaled(model, values, batch_size)
    prediction = prepared.inverse_predictions(output["prediction"])
    complete = evaluation_table(
        split.y_raw,
        prediction,
        quantiles=model.config.quantiles,
        mask=split.target_mask,
        include_sides=include_sides,
    )
    requested = tuple(metrics)
    unknown = set(requested).difference(complete.columns)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    selected = complete.loc[:, ("Scope", *requested)]

    forecasts = {
        "prediction": prediction.astype(np.float32),
        "prediction_scaled": output["prediction"].astype(np.float32),
        "target": split.y_raw.astype(np.float32),
        "target_mask": split.target_mask.astype(np.uint8),
        "buy_action": output["buy_action"].astype(np.int16),
        "sell_action": output["sell_action"].astype(np.int16),
        "buy_probability": output["buy_probability"].astype(np.float32),
        "sell_probability": output["sell_probability"].astype(np.float32),
        "delivery_time_utc_ns": pd.DatetimeIndex(split.meta[DATE_COLUMN]).asi8,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        complete.to_csv(destination / f"{split_name}_metrics_all.csv", index=False)
        selected.to_csv(destination / f"{split_name}_metrics.csv", index=False)
        np.savez_compressed(
            destination / f"{split_name}_forecasts.npz", **forecasts
        )
    return selected, forecasts


def evaluation_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
    include_sides: bool = False,
) -> pd.DataFrame:
    """Return probabilistic, point, negative-price, and side metrics."""

    y_true, y_pred, valid, quantiles = _validated_inputs(
        y_true, y_pred, quantiles, mask
    )
    overall = _metric_row("Overall", y_true, y_pred, valid, quantiles)
    overall["ASCR"] = average_side_crossing_rate(y_pred, valid)
    rows = [overall]
    if include_sides:
        for side_index, side_name in enumerate(SIDE_NAMES[: y_true.shape[-1]]):
            rows.append(
                _metric_row(
                    side_name,
                    np.take(y_true, [side_index], axis=-1),
                    np.take(y_pred, [side_index], axis=-2),
                    np.take(valid, [side_index], axis=-1),
                    quantiles,
                )
            )
    return pd.DataFrame(rows)


def average_quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> float:
    y_true, y_pred, valid, quantiles = _validated_inputs(
        y_true, y_pred, quantiles, mask
    )
    loss = _pinball(y_true, y_pred, quantiles)
    valid_quantiles = np.broadcast_to(valid[..., None], loss.shape)
    return float(loss[valid_quantiles].mean()) if valid_quantiles.any() else math.nan


def coverage_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> dict[float, float]:
    y_true, y_pred, valid, quantiles = _validated_inputs(
        y_true, y_pred, quantiles, mask
    )
    output = {}
    for index, quantile in enumerate(quantiles):
        keep = valid & np.isfinite(y_pred[..., index])
        output[quantile] = (
            float(np.mean(y_true[keep] <= y_pred[..., index][keep]))
            if keep.any()
            else math.nan
        )
    return output


def average_quantile_coverage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> float:
    coverage = coverage_rates(y_true, y_pred, quantiles=quantiles, mask=mask)
    return float(np.mean([abs(coverage[q] - q) for q in coverage]))


def quantile_crossing_rate(
    y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    prediction = np.asarray(y_pred, dtype=np.float64)
    if prediction.ndim < 3:
        raise ValueError("y_pred must end with a quantile axis")
    valid = np.all(np.isfinite(prediction), axis=-1)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != valid.shape:
            raise ValueError("mask must match all axes except quantiles")
        valid &= supplied
    crossing = np.diff(prediction, axis=-1) < 0.0
    valid_pairs = np.broadcast_to(valid[..., None], crossing.shape)
    return float(crossing[valid_pairs].mean()) if valid_pairs.any() else math.nan


def mean_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    y_true, y_pred, valid = _validated_points(y_true, y_pred, mask)
    return float(np.abs(y_true[valid] - y_pred[valid]).mean()) if valid.any() else math.nan


def root_mean_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    y_true, y_pred, valid = _validated_points(y_true, y_pred, mask)
    return (
        float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))
        if valid.any()
        else math.nan
    )


def r2_score(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    y_true, y_pred, valid = _validated_points(y_true, y_pred, mask)
    if not valid.any():
        return math.nan
    actual = y_true[valid]
    residual_sum = float(np.sum((actual - y_pred[valid]) ** 2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - residual_sum / total_sum if total_sum > 0.0 else math.nan


def negative_price_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    y_true, y_pred, valid, quantiles = _validated_inputs(
        y_true, y_pred, quantiles, mask
    )
    median = y_pred[..., _quantile_index(quantiles, 0.5)]
    actual = y_true[valid] < 0.0
    predicted = median[valid] < 0.0
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision + recall) and precision + recall > 0.0
        else math.nan
    )
    return {"Precision": precision, "Recall": recall, "F1": f1}


def average_side_crossing_rate(
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    buy_index: int = 0,
    sell_index: int = 1,
) -> float:
    prediction = np.asarray(y_pred, dtype=np.float64)
    if prediction.ndim < 4 or prediction.shape[-2] < 2:
        return math.nan
    buy = prediction[..., buy_index, :]
    sell = prediction[..., sell_index, :]
    valid = np.isfinite(buy) & np.isfinite(sell)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != prediction.shape[:-1]:
            raise ValueError("mask must match all axes except quantiles")
        valid &= (supplied[..., buy_index] & supplied[..., sell_index])[..., None]
    return float((buy[valid] < sell[valid]).mean()) if valid.any() else math.nan


def metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("origin",),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    if len(meta) != len(y_true):
        raise ValueError("meta and forecasts must have the same sample count")
    rows = []
    groups = meta.reset_index(drop=True).groupby(list(group_columns), sort=True).indices
    for key, indices in groups.items():
        indices = np.asarray(indices, dtype=int)
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        result = evaluation_table(
            np.asarray(y_true)[indices],
            np.asarray(y_pred)[indices],
            quantiles=quantiles,
            mask=np.asarray(mask)[indices] if mask is not None else None,
        ).iloc[0]
        row.update(result.drop(labels="Scope").to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def losses_per_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "AQL",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    y_true, y_pred, valid, quantiles = _validated_inputs(
        y_true, y_pred, quantiles, mask
    )
    metric = metric.upper()
    if metric == "AQL":
        values = _pinball(y_true, y_pred, quantiles)
        valid_values = np.broadcast_to(valid[..., None], values.shape)
    else:
        median = y_pred[..., _quantile_index(quantiles, 0.5)]
        error = y_true - median
        if metric == "MAE":
            values = np.abs(error)
        elif metric in {"MSE", "RMSE"}:
            values = error**2
        else:
            raise ValueError("metric must be AQL, MAE, MSE, or RMSE")
        valid_values = valid & np.isfinite(values)
    output = _row_mean(values, valid_values)
    return np.sqrt(output) if metric == "RMSE" else output


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    newey_west_lag: int | None = None,
) -> pd.DataFrame:
    """Two-sided Harvey-Leybourne-Newbold-corrected DM test."""

    loss_a = np.asarray(loss_a, dtype=np.float64).reshape(-1)
    loss_b = np.asarray(loss_b, dtype=np.float64).reshape(-1)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss arrays must have the same shape")
    valid = np.isfinite(loss_a) & np.isfinite(loss_b)
    differential = loss_a[valid] - loss_b[valid]
    sample_count = len(differential)
    if sample_count < 3:
        raise ValueError("DM test requires at least three common observations")
    lag = (
        automatic_newey_west_lag(sample_count)
        if newey_west_lag is None
        else int(newey_west_lag)
    )
    if lag < 0 or lag >= sample_count:
        raise ValueError("newey_west_lag must be between zero and n-1")
    mean_difference = float(differential.mean())
    centered = differential - mean_difference
    long_run_variance = float(np.dot(centered, centered) / sample_count)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / sample_count
        )
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(
        max(long_run_variance / sample_count, np.finfo(float).tiny)
    )
    statistic = mean_difference / standard_error
    correction = (
        sample_count
        + 1
        - 2 * horizon
        + horizon * (horizon - 1) / sample_count
    ) / sample_count
    statistic *= math.sqrt(max(correction, 0.0))
    return pd.DataFrame(
        [
            {
                "N": sample_count,
                "MeanDifferenceAminusB": mean_difference,
                "DMStatistic": statistic,
                "PValue": math.erfc(abs(statistic) / math.sqrt(2.0)),
                "BetterForecast": "A" if mean_difference < 0.0 else "B",
            }
        ]
    )


def compare_forecasts_dm(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    metric: str = "AQL",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    mask: np.ndarray | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    loss_a = losses_per_sample(
        y_true, prediction_a, metric=metric, quantiles=quantiles, mask=mask
    )
    loss_b = losses_per_sample(
        y_true, prediction_b, metric=metric, quantiles=quantiles, mask=mask
    )
    result = diebold_mariano_test(loss_a, loss_b, horizon=horizon)
    result.insert(0, "Metric", metric.upper())
    return result


def automatic_newey_west_lag(sample_count: int) -> int:
    return max(
        0,
        min(
            sample_count - 1,
            int(math.floor(4.0 * (sample_count / 100.0) ** (2.0 / 9.0))),
        ),
    )


def _metric_row(
    scope: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[str, float | int | str]:
    coverage = coverage_rates(y_true, y_pred, quantiles=quantiles, mask=valid)
    median = y_pred[..., _quantile_index(quantiles, 0.5)]
    row: dict[str, float | int | str] = {
        "Scope": scope,
        "AQL": average_quantile_loss(
            y_true, y_pred, quantiles=quantiles, mask=valid
        ),
        "AQCE": float(np.mean([abs(coverage[q] - q) for q in quantiles])),
        "AQCR": quantile_crossing_rate(y_pred, valid),
        "MAE": mean_absolute_error(y_true, median, valid),
        "RMSE": root_mean_squared_error(y_true, median, valid),
        "R2": r2_score(y_true, median, valid),
        "NTargets": int(valid.sum()),
        "ASCR": math.nan,
    }
    for quantile, value in coverage.items():
        row[f"Coverage_q{int(round(100 * quantile)):02d}"] = value
    row.update(negative_price_scores(y_true, y_pred, quantiles=quantiles, mask=valid))
    return row


def _validated_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Sequence[float],
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    quantiles = tuple(float(value) for value in quantiles)
    if y_pred.shape != (*y_true.shape, len(quantiles)):
        raise ValueError("y_pred must have shape y_true.shape + [quantiles]")
    if tuple(sorted(quantiles)) != quantiles or any(
        value <= 0.0 or value >= 1.0 for value in quantiles
    ):
        raise ValueError("quantiles must be increasing and inside (0, 1)")
    valid = np.isfinite(y_true) & np.all(np.isfinite(y_pred), axis=-1)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true")
        valid &= supplied
    return y_true, y_pred, valid, quantiles


def _validated_points(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true")
        valid &= supplied
    return y_true, y_pred, valid


def _pinball(
    y_true: np.ndarray, y_pred: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    quantile = np.asarray(tuple(quantiles)).reshape(
        (1,) * (y_pred.ndim - 1) + (len(tuple(quantiles)),)
    )
    error = y_true[..., None] - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def _quantile_index(quantiles: Sequence[float], requested: float) -> int:
    matches = np.flatnonzero(np.isclose(np.asarray(tuple(quantiles)), requested))
    if not len(matches):
        raise ValueError(f"quantiles must contain {requested}")
    return int(matches[0])


def _row_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, values.ndim))
    numerator = np.where(valid, values, 0.0).sum(axis=axes)
    denominator = valid.sum(axis=axes)
    output = np.full(len(values), np.nan, dtype=np.float64)
    usable = denominator > 0
    output[usable] = numerator[usable] / denominator[usable]
    return output


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


evaluate_model = evaluate_orderfusion_plus
