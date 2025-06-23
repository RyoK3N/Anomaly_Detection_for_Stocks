"""Technical indicator utilities."""

from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with simple moving averages and RSI.

    Args:
        df: Input dataframe with at least a ``Close`` column.

    Returns:
        ``pd.DataFrame`` with ``MA20``, ``MA50`` and ``RSI`` columns appended.
    """

    result = df.copy()
    result["MA20"] = result["Close"].rolling(window=20).mean()
    result["MA50"] = result["Close"].rolling(window=50).mean()

    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    result["RSI"] = 100 - (100 / (1 + rs))
    result.fillna(method="bfill", inplace=True)
    return result
