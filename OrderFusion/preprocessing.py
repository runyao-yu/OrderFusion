"""Reusable loading, splitting, scaling, and tensor construction for OrderFusion+."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DATE_COLUMN = "Delivery start time"
DELTA_COLUMN = "TimeDelta"
SIDES = ("buy", "sell")
CALENDAR_FEATURE_NAMES = (
    "delivery_time_sin",
    "delivery_time_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_of_year_sin",
    "month_of_year_cos",
    "is_holiday",
)
SUPPORTED_FEATURES = (
    "VWAP",
    "Traded_volume",
    "Number_of_trades",
    "TimeDelta",
    "Min_price",
    "Max_price",
)


@dataclass(frozen=True)
class SplitDates:
    """Half-open delivery-time ranges. Starts are included and ends excluded."""

    train_start: str | pd.Timestamp
    train_end: str | pd.Timestamp
    validation_start: str | pd.Timestamp
    validation_end: str | pd.Timestamp
    test_start: str | pd.Timestamp
    test_end: str | pd.Timestamp

    def ranges(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        ranges = {
            "train": (_utc(self.train_start), _utc(self.train_end)),
            "validation": (
                _utc(self.validation_start),
                _utc(self.validation_end),
            ),
            "test": (_utc(self.test_start), _utc(self.test_end)),
        }
        for name, (start, end) in ranges.items():
            if start >= end:
                raise ValueError(f"{name}_start must be earlier than {name}_end")
        ordered = sorted((start, end, name) for name, (start, end) in ranges.items())
        for (_, previous_end, previous_name), (next_start, _, next_name) in zip(
            ordered, ordered[1:]
        ):
            if next_start < previous_end:
                raise ValueError(
                    f"The {previous_name} and {next_name} ranges overlap"
                )
        return ranges


@dataclass(frozen=True)
class ScalerParameters:
    scaler_type: str
    transform: str
    center: float
    scale: float

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        values = _apply_transform(np.asarray(values, dtype=np.float32), self.transform)
        values = (values - self.center) / self.scale
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def inverse_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32) * self.scale + self.center
        if self.transform == "identity":
            return values
        if self.transform == "log1p":
            return np.expm1(values)
        raise ValueError(f"Unknown transform: {self.transform}")


@dataclass
class DataSplit:
    buy: np.ndarray
    sell: np.ndarray
    buy_missing: np.ndarray
    sell_missing: np.ndarray
    context: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    target_mask: np.ndarray
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.buy)


@dataclass
class PreparedData:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    feature_names: tuple[str, ...]
    input_scalers: dict[str, ScalerParameters]
    target_scalers: dict[str, ScalerParameters]
    settings: dict[str, object]

    @property
    def timesteps(self) -> int:
        return int(self.train.buy.shape[1])

    @property
    def products(self) -> int:
        return int(self.train.buy.shape[2])

    @property
    def input_channels(self) -> int:
        return int(self.train.buy.shape[3])

    @property
    def context_features(self) -> int:
        return int(self.train.context.shape[1])

    @property
    def output_steps(self) -> int:
        return int(self.train.y_scaled.shape[1])

    def inverse_predictions(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.ndim != 4 or prediction.shape[2] != len(SIDES):
            raise ValueError(
                "prediction must have shape [samples, output_steps, 2, quantiles]"
            )
        output = prediction.copy()
        for side_index, side in enumerate(SIDES):
            output[:, :, side_index, :] = self.target_scalers[side].inverse_values(
                output[:, :, side_index, :]
            )
        return output

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "validation", "test"):
            split = getattr(self, name)
            rows.append(
                {
                    "split": name,
                    "samples": len(split),
                    "first_delivery": split.meta[DATE_COLUMN].min(),
                    "last_delivery": split.meta[DATE_COLUMN].max(),
                    "observed_target_rate": float(split.target_mask.mean()),
                    "input_shape": str(tuple(split.buy.shape)),
                    "target_shape": str(tuple(split.y_scaled.shape)),
                }
            )
        return pd.DataFrame(rows)


def load_processed_data(
    path: str | Path, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load one processed parquet file and validate its event-table index."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    missing = {DATE_COLUMN, DELTA_COLUMN}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="raise")
    frame[DELTA_COLUMN] = pd.to_numeric(frame[DELTA_COLUMN], errors="raise").astype(
        int
    )
    return frame


def history_deltas(
    origin: int, input_length_minutes: int, bucket_minutes: int
) -> np.ndarray:
    """Oldest-to-newest bucket labels available at the forecast origin."""

    if origin >= 0:
        raise ValueError("origin must be negative")
    if input_length_minutes <= 0 or input_length_minutes % bucket_minutes:
        raise ValueError("input length must be a positive bucket multiple")
    steps = input_length_minutes // bucket_minutes
    latest = abs(origin) + bucket_minutes
    oldest = latest + (steps - 1) * bucket_minutes
    return np.arange(oldest, latest - bucket_minutes, -bucket_minutes, dtype=np.int32)


def trajectory_deltas(origin: int, bucket_minutes: int) -> np.ndarray:
    """Chronological target buckets from the forecast origin to delivery."""

    if origin >= 0 or abs(origin) % bucket_minutes:
        raise ValueError("origin must be negative and divisible by bucket_minutes")
    return np.arange(abs(origin), 0, -bucket_minutes, dtype=np.int32)


def make_calendar_context(
    dates: pd.DatetimeIndex,
    *,
    country: str,
    local_timezone: str = "Europe/Berlin",
) -> np.ndarray:
    """Return seven periodic calendar features in local market time."""

    local = pd.DatetimeIndex(dates).tz_convert(local_timezone)
    minute = (
        local.hour.to_numpy(dtype=np.float32) * 60.0
        + local.minute.to_numpy(dtype=np.float32)
    )
    weekday = local.dayofweek.to_numpy(dtype=np.float32)
    month = local.month.to_numpy(dtype=np.float32)
    holidays = {
        int(year): national_holidays(country.upper(), int(year))
        for year in np.unique(local.year)
    }
    holiday = np.asarray(
        [timestamp.date() in holidays[timestamp.year] for timestamp in local],
        dtype=np.float32,
    )
    return np.column_stack(
        (
            np.sin(2.0 * np.pi * minute / 1440.0),
            np.cos(2.0 * np.pi * minute / 1440.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            holiday,
        )
    ).astype(np.float32)


def prepare_orderfusion_data(
    frame: pd.DataFrame,
    *,
    country: str,
    bucket_minutes: int,
    product_minutes: int,
    origins: Sequence[int],
    max_input_minutes: int,
    max_neighbors: int,
    split_dates: SplitDates,
    feature_names: Sequence[str] = ("VWAP", "Traded_volume"),
    scaler_type: str = "standard",
    feature_transforms: Mapping[str, str] | None = None,
    local_timezone: str = "Europe/Berlin",
) -> PreparedData:
    """Create delivery-aligned BUY/SELL tensors and padded trajectories."""

    country = country.upper()
    features = tuple(feature_names)
    _validate_settings(
        frame,
        country,
        bucket_minutes,
        product_minutes,
        origins,
        max_input_minutes,
        max_neighbors,
        features,
        scaler_type,
    )
    ranges = split_dates.ranges()
    transforms = {
        "VWAP": "identity",
        "Traded_volume": "log1p",
        "Number_of_trades": "log1p",
        "TimeDelta": "identity",
        "Min_price": "identity",
        "Max_price": "identity",
        **dict(feature_transforms or {}),
    }

    needed_columns = {DATE_COLUMN, DELTA_COLUMN}
    for side in SIDES:
        needed_columns.update((f"Missing_value_{side}", f"VWAP_{side}"))
        for feature in features:
            if feature != "TimeDelta":
                needed_columns.add(f"{feature}_{side}")
    missing_columns = needed_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing processed-data columns: {sorted(missing_columns)}")

    frame = frame.loc[:, sorted(needed_columns)].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="raise")
    frame[DELTA_COLUMN] = pd.to_numeric(frame[DELTA_COLUMN], errors="raise").astype(
        int
    )
    if frame.duplicated([DATE_COLUMN, DELTA_COLUMN]).any():
        raise ValueError("Delivery start time and TimeDelta must identify unique rows")

    train_start, train_end = ranges["train"]
    scaler_frame = frame[
        (frame[DATE_COLUMN] >= train_start) & (frame[DATE_COLUMN] < train_end)
    ]
    if scaler_frame.empty:
        raise ValueError("No source rows fall inside the training range")

    input_scalers: dict[str, ScalerParameters] = {}
    for side in SIDES:
        observed = (
            pd.to_numeric(scaler_frame[f"Missing_value_{side}"], errors="coerce")
            .fillna(0)
            .eq(1)
            .to_numpy()
        )
        for feature in features:
            source = DELTA_COLUMN if feature == "TimeDelta" else f"{feature}_{side}"
            values = pd.to_numeric(scaler_frame[source], errors="coerce").to_numpy(
                dtype=np.float64
            )
            input_scalers[f"{side}:{feature}"] = fit_scaler(
                values[observed], scaler_type, transforms[feature]
            )

    target_scalers: dict[str, ScalerParameters] = {}
    for side in SIDES:
        key = f"{side}:VWAP"
        if key in input_scalers:
            target_scalers[side] = input_scalers[key]
        else:
            observed = (
                pd.to_numeric(scaler_frame[f"Missing_value_{side}"], errors="coerce")
                .fillna(0)
                .eq(1)
                .to_numpy()
            )
            values = pd.to_numeric(
                scaler_frame[f"VWAP_{side}"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            target_scalers[side] = fit_scaler(values[observed], scaler_type)

    histories = {
        int(origin): history_deltas(int(origin), max_input_minutes, bucket_minutes)
        for origin in origins
    }
    max_origin_minutes = max(abs(int(origin)) for origin in origins)
    output_deltas = np.arange(
        max_origin_minutes, 0, -bucket_minutes, dtype=np.int32
    )
    target_deltas = {
        int(origin): trajectory_deltas(int(origin), bucket_minutes)
        for origin in origins
    }
    required_deltas = set(output_deltas.tolist())
    for base_deltas in histories.values():
        for neighbor in range(max_neighbors + 1):
            required_deltas.update(
                (base_deltas + neighbor * product_minutes).tolist()
            )
    unavailable = sorted(required_deltas.difference(frame[DELTA_COLUMN].unique()))
    if unavailable:
        raise ValueError(
            f"Processed data lack {len(unavailable)} required TimeDelta values; "
            f"first values: {unavailable[:12]}"
        )

    value_columns = sorted(needed_columns.difference({DATE_COLUMN, DELTA_COLUMN}))
    reduced = frame[frame[DELTA_COLUMN].isin(required_deltas)]
    wide = reduced.pivot(
        index=DATE_COLUMN, columns=DELTA_COLUMN, values=value_columns
    ).sort_index()
    complete_index = pd.date_range(
        wide.index.min(), wide.index.max(), freq=f"{product_minutes}min", tz="UTC"
    )
    wide = wide.reindex(complete_index)
    last_usable = wide.index.max() - pd.Timedelta(
        minutes=max_neighbors * product_minutes
    )
    eligible_mask = np.zeros(len(wide.index), dtype=bool)
    for start, end in ranges.values():
        eligible_mask |= np.asarray((wide.index >= start) & (wide.index < end))
    eligible = wide.index[eligible_mask & np.asarray(wide.index <= last_usable)]
    if len(eligible) == 0:
        raise ValueError("No products are eligible in the requested split ranges")

    chunk_names = (
        "buy",
        "sell",
        "buy_missing",
        "sell_missing",
        "y_scaled",
        "y_raw",
        "target_mask",
        "date",
        "origin",
        "target_steps",
    )
    chunks: dict[str, list] = {name: [] for name in chunk_names}

    for origin in map(int, origins):
        base_deltas = histories[origin]
        side_products: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        mask_products: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}

        for neighbor in range(max_neighbors, -1, -1):
            product_dates = eligible + pd.Timedelta(
                minutes=neighbor * product_minutes
            )
            deltas = base_deltas + neighbor * product_minutes
            product = wide.reindex(product_dates)
            for side in SIDES:
                missing = _wide_values(product, f"Missing_value_{side}", deltas)
                missing = np.clip(np.nan_to_num(missing), 0.0, 1.0).astype(
                    np.float32
                )
                channels = []
                for feature in features:
                    if feature == "TimeDelta":
                        raw = np.broadcast_to(
                            deltas.reshape(1, -1), (len(product), len(deltas))
                        ).astype(np.float32)
                    else:
                        raw = _wide_values(product, f"{feature}_{side}", deltas)
                    scaled = input_scalers[f"{side}:{feature}"].transform_values(raw)
                    channels.append(scaled * missing)
                side_products[side].append(np.stack(channels, axis=-1))
                mask_products[side].append(missing)

        for side in SIDES:
            chunks[side].append(
                np.stack(side_products[side], axis=2).astype(np.float32)
            )
            chunks[f"{side}_missing"].append(
                np.stack(mask_products[side], axis=2)[..., None].astype(np.float32)
            )

        target = wide.reindex(eligible)
        y_raw = np.stack(
            [
                _wide_values(target, f"VWAP_{side}", output_deltas)
                for side in SIDES
            ],
            axis=2,
        ).astype(np.float32)
        target_mask = np.stack(
            [
                _wide_values(target, f"Missing_value_{side}", output_deltas)
                for side in SIDES
            ],
            axis=2,
        )
        target_mask *= (output_deltas <= abs(origin)).reshape(1, -1, 1)
        target_mask = np.clip(np.nan_to_num(target_mask), 0.0, 1.0)
        target_mask *= np.isfinite(y_raw)
        y_raw = np.where(target_mask.astype(bool), y_raw, np.nan).astype(np.float32)
        y_scaled = np.zeros_like(y_raw, dtype=np.float32)
        for side_index, side in enumerate(SIDES):
            values = target_scalers[side].transform_values(y_raw[:, :, side_index])
            y_scaled[:, :, side_index] = values * target_mask[:, :, side_index]

        chunks["y_raw"].append(y_raw)
        chunks["y_scaled"].append(y_scaled)
        chunks["target_mask"].append(target_mask.astype(np.float32))
        chunks["date"].append(np.asarray(eligible))
        chunks["origin"].append(np.full(len(eligible), origin, dtype=np.int16))
        chunks["target_steps"].append(
            np.full(len(eligible), len(target_deltas[origin]), dtype=np.int16)
        )

    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    dates = pd.DatetimeIndex(arrays.pop("date"))
    arrays["context"] = make_calendar_context(
        dates, country=country, local_timezone=local_timezone
    )
    meta = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "origin": arrays["origin"],
            "target_steps": arrays["target_steps"],
        }
    )

    split_objects: dict[str, DataSplit] = {}
    for name, (start, end) in ranges.items():
        index = np.flatnonzero(np.asarray((dates >= start) & (dates < end)))
        if len(index) == 0:
            raise ValueError(f"The {name} split has no samples")
        split_objects[name] = DataSplit(
            buy=arrays["buy"][index],
            sell=arrays["sell"][index],
            buy_missing=arrays["buy_missing"][index],
            sell_missing=arrays["sell_missing"][index],
            context=arrays["context"][index],
            y_scaled=arrays["y_scaled"][index],
            y_raw=arrays["y_raw"][index],
            target_mask=arrays["target_mask"][index],
            meta=meta.iloc[index].reset_index(drop=True),
        )

    settings = {
        "country": country,
        "bucket_minutes": int(bucket_minutes),
        "product_minutes": int(product_minutes),
        "origins": tuple(map(int, origins)),
        "max_input_minutes": int(max_input_minutes),
        "max_neighbors": int(max_neighbors),
        "output_deltas": tuple(map(int, output_deltas)),
        "feature_names": features,
        "scaler_type": scaler_type,
        "split_dates": asdict(split_dates),
        "context_names": CALENDAR_FEATURE_NAMES,
    }
    return PreparedData(
        train=split_objects["train"],
        validation=split_objects["validation"],
        test=split_objects["test"],
        feature_names=features,
        input_scalers=input_scalers,
        target_scalers=target_scalers,
        settings=settings,
    )


def fit_scaler(
    values: np.ndarray, scaler_type: str, transform: str = "identity"
) -> ScalerParameters:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot fit a scaler without finite training values")
    values = _apply_transform(values, transform)
    values = values[np.isfinite(values)]
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        center, scale = float(values.mean()), float(values.std())
    elif scaler_type == "robust":
        center = float(np.median(values))
        lower, upper = np.percentile(values, (25.0, 75.0))
        scale = float(upper - lower)
    elif scaler_type in {"none", "identity"}:
        center, scale, scaler_type = 0.0, 1.0, "none"
    else:
        raise ValueError("scaler_type must be 'standard', 'robust', or 'none'")
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return ScalerParameters(scaler_type, transform, center, scale)


def national_holidays(country: str, year: int) -> set[date]:
    easter = _easter_sunday(year)
    common = {
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
        easter + timedelta(days=1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
    }
    if country == "DE":
        return common | {date(year, 10, 3), easter - timedelta(days=2)}
    if country == "AT":
        return common | {
            date(year, 1, 6),
            date(year, 8, 15),
            date(year, 10, 26),
            date(year, 11, 1),
            date(year, 12, 8),
            easter + timedelta(days=60),
        }
    raise ValueError("country must be 'DE' or 'AT'")


def _validate_settings(
    frame: pd.DataFrame,
    country: str,
    bucket_minutes: int,
    product_minutes: int,
    origins: Sequence[int],
    max_input_minutes: int,
    max_neighbors: int,
    features: Sequence[str],
    scaler_type: str,
) -> None:
    missing = {DATE_COLUMN, DELTA_COLUMN}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if country not in {"DE", "AT"}:
        raise ValueError("country must be 'DE' or 'AT'")
    if bucket_minutes <= 0 or product_minutes <= 0:
        raise ValueError("bucket_minutes and product_minutes must be positive")
    if max_input_minutes <= 0 or max_input_minutes % bucket_minutes:
        raise ValueError("max_input_minutes must be a positive bucket multiple")
    if max_neighbors < 0:
        raise ValueError("max_neighbors cannot be negative")
    if not origins or any(int(origin) >= 0 for origin in origins):
        raise ValueError("origins must contain negative values")
    if any(abs(int(origin)) % bucket_minutes for origin in origins):
        raise ValueError("every origin magnitude must be a bucket multiple")
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_names must contain unique values")
    unsupported = set(features).difference(SUPPORTED_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported features: {sorted(unsupported)}")
    if scaler_type.lower() not in {"standard", "robust", "none", "identity"}:
        raise ValueError("scaler_type must be 'standard', 'robust', or 'none'")


def _wide_values(
    frame: pd.DataFrame, value_column: str, deltas: Sequence[int]
) -> np.ndarray:
    columns = pd.MultiIndex.from_tuples(
        [(value_column, int(delta)) for delta in deltas],
        names=frame.columns.names,
    )
    return frame.reindex(columns=columns).to_numpy(dtype=np.float32)


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values)
    if transform == "identity":
        return values
    if transform == "log1p":
        finite = values[np.isfinite(values)]
        if finite.size and np.any(finite < 0):
            raise ValueError("log1p requires non-negative values")
        return np.log1p(values)
    raise ValueError("transform must be 'identity' or 'log1p'")


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


prepare_data = prepare_orderfusion_data
