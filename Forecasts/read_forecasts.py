"""Read the OrderFusion forecasts file (orderfusion_forecasts_2024.npz).

The file holds the probabilistic buy-sell VWAP trajectory forecasts of OrderFusion+ and
all baselines for the German continuous intraday market (15-min products, test year 2024,
three test folds merged). It contains derived information only: model forecasts. No realised
prices, no volumes, no raw orderbook data.

Only numpy is required (pandas is optional, for DataFrames).

    from read_forecasts import Forecasts
    f = Forecasts("orderfusion_forecasts_2024.npz")
    f.models                                   # available models
    f.get("OrderFusionPlus", "2024-07-23 18:00", origin=-180)   # one delivery product
    f.day("LSTM", "2024-07-23", origin=-60)    # all 96 products of one day, array [96, 4, 2, 3]
    f.metrics["OrderFusionPlus"]["-180"]       # AQL, AQCE, MAE, R2 on the 2024 test year

All times are delivery start times in UTC. Prices are in EUR/MWh.
A bucket labelled d covers the trades from d to d-15 minutes before delivery start.

File layout (a standard zip / .npz, every member can be read on its own):
    meta_json            uint8, JSON with models, origins, quantiles, scales, metrics
    <model>/<YYYY-MM-DD> int16 [96, 24, 2, 3]   encoded forecasts * 10, see _decode()
The 24 forecast steps are the three origins one after another:
    origin -180: 12 steps, origin -120: 8 steps, origin -60: 4 steps.
"""
from __future__ import annotations

import json
import sys

import numpy as np


class Forecasts:
    def __init__(self, path: str = "orderfusion_forecasts_2024.npz"):
        self._z = np.load(path)
        self.meta = json.loads(self._z["meta_json"].tobytes().decode())
        self.models = self.meta["models"]
        self.origins = self.meta["origins"]
        self.quantiles = self.meta["quantiles"]
        self.days = self.meta["days"]
        self.metrics = self.meta["metrics"]
        edges = np.cumsum([0] + self.meta["steps_per_origin"])
        self._block = {o: (int(edges[i]), int(edges[i + 1])) for i, o in enumerate(self.origins)}

    # ------------------------------------------------------------------ forecasts
    def _decode(self, enc: np.ndarray) -> np.ndarray:
        """Undo the storage transform. Stored per origin block:
        step differences of (q10 - q50, q50, q90 - q50), in 0.1 EUR/MWh."""
        missing = enc[:, 0, 0, 1] == np.iinfo(enc.dtype).min
        v = enc.astype(np.int64)
        for a, b in self._block.values():
            v[:, a:b] = np.cumsum(v[:, a:b], axis=1)
        v[..., 0] += v[..., 1]
        v[..., 2] += v[..., 1]
        out = v / self.meta["forecast_scale"]
        out[missing] = np.nan
        return out

    def day(self, model: str, date: str, origin: int | None = None) -> np.ndarray:
        """Forecasts of all 96 quarter-hour products of one UTC day.
        Returns [96, steps, side(buy, sell), quantile(0.1, 0.5, 0.9)]; NaN if not available."""
        out = self._decode(self._z[f"{model}/{date}"])
        if origin is None:
            return out
        a, b = self._block[int(origin)]
        return out[:, a:b]

    @staticmethod
    def _split(delivery: str) -> tuple[str, int]:
        ts = np.datetime64(delivery.replace("T", " ").replace("Z", ""), "m")
        day = ts.astype("datetime64[D]")
        return str(day), int((ts - day) / np.timedelta64(15, "m"))

    def get(self, model: str, delivery: str, origin: int = -180):
        """Forecast of one product from one forecasting origin."""
        date, slot = self._split(delivery)
        pred = self.day(model, date, origin)[slot]
        steps = pred.shape[0]
        cols = {"minutes_before_delivery": list(range(15 * steps, 0, -15))}
        for s, side in enumerate(self.meta["sides"]):
            for q, quantile in enumerate(self.quantiles):
                cols[f"{side}_q{quantile}"] = pred[:, s, q]
        return _frame(cols)

    def year(self, model: str, origin: int = -180):
        """All forecasts of one model and origin: (delivery_times, forecasts)."""
        pred = np.concatenate([self.day(model, d, origin) for d in self.days])
        start = np.datetime64(self.days[0], "m")
        t = start + np.arange(len(pred)) * np.timedelta64(15, "m")
        keep = np.isfinite(pred).all(axis=(1, 2, 3))
        return t[keep], pred[keep]


def _frame(columns: dict):
    try:
        import pandas as pd
        return pd.DataFrame(columns)
    except ImportError:
        return columns


if __name__ == "__main__":
    # python read_forecasts.py OrderFusionPlus "2024-07-23 18:00" -180
    f = Forecasts()
    if len(sys.argv) < 3:
        print("models:", ", ".join(f.models))
        print('usage: python read_forecasts.py <model> "<YYYY-MM-DD HH:MM>" [origin]')
    else:
        print(f.get(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else -180))
