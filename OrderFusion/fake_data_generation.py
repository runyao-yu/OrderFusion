"""Fake input data for OrderFusion+.

`make_fake_data` builds a table of random numbers with exactly the layout of the processed
trade-feature file that `preprocessing.load_processed_data` reads. It contains no market data.
Use it to see the required data structure, or to check that your installation runs end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_fake_data(start='2024-01-01', end='2024-01-04', product_minutes=15,
                   bucket_minutes=15, max_delta=720, seed=0):
    """Return one row per delivery product and 15-minute bucket.

    Columns: `Delivery start time` (UTC), `TimeDelta` (minutes before delivery; d covers the trades
    from d to d-15 minutes before delivery start), and per side (buy, sell) `VWAP_<side>` in EUR/MWh
    (NaN without trades), `Traded_volume_<side>` in MWh (0 without trades) and `Missing_value_<side>`
    (1 if the bucket has at least one trade, else 0).
    """
    rng = np.random.default_rng(seed)
    deliveries = pd.date_range(start, end, freq=f'{product_minutes}min', tz='UTC', inclusive='left')
    deltas = np.arange(max_delta, 0, -bucket_minutes)             # minutes before delivery, oldest first
    rows = []
    for delivery in deliveries:
        level = 80 + 30 * np.sin(2 * np.pi * (delivery.hour - 7) / 24) + rng.normal(0, 5)
        mid = level + np.cumsum(rng.normal(0, 3, len(deltas)))     # random-walk price path
        frame = pd.DataFrame({'Delivery start time': delivery, 'TimeDelta': deltas})
        for side, shift in (('buy', 0.5), ('sell', -0.5)):
            traded = rng.random(len(deltas)) < np.linspace(0.5, 1.0, len(deltas))   # more trades close to delivery
            frame[f'VWAP_{side}'] = np.where(traded, mid + shift + rng.normal(0, 1, len(deltas)), np.nan)
            frame[f'Traded_volume_{side}'] = np.where(traded, rng.gamma(2.0, 10.0, len(deltas)), 0.0)
            frame[f'Missing_value_{side}'] = traded.astype(int)     # 1 = at least one trade in the bucket
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
