from typing import List, Union

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_finance_df_with_symbol(symbol: str) -> pd.DataFrame:
    """
    Fetch 1-year daily historical price data for a given stock symbol.

    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        pd.DataFrame: OHLCV data for the past year at 1-day intervals.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y", interval="1d")
    return df


def last(arr: Union[List[float], np.ndarray]) -> float:
    """
    Return the last element of an array as a float.

    Args:
        arr (list | np.ndarray): Input numeric array.

    Returns:
        float: Last value in the array, or 0.0 if empty.
    """
    return float(arr[-1]) if len(arr) else 0.0


def minmax01(x: Union[List[float], np.ndarray]) -> List[float]:
    """
    Apply Min-Max normalization to a numeric sequence (0-1 scaling).

    Args:
        x (list | np.ndarray): Input numeric sequence.

    Returns:
        list[float]: Normalized list in [0, 1].
    """
    arr = np.array(x, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return np.zeros_like(arr, dtype=float).tolist()
    return ((arr - mn) / (mx - mn)).clip(0, 1).tolist()


def tail(s: pd.Series, k: int = 14, d: int = 3) -> List[float]:
    """
    Extract the last k non-NaN values from a Series and round them.

    Args:
        s (pd.Series): Input numeric Series.
        k (int): Number of tail values to keep. Defaults to 14.
        d (int): Decimal rounding. Defaults to 3.

    Returns:
        list[float]: Last k rounded values.
    """
    return s.dropna().tail(k).round(d).tolist()


def to_float_list(xs: List[Union[int, float]], nd: int = 3) -> List[float]:
    """
    Convert a numeric sequence to a list of floats rounded to nd decimals.

    Args:
        xs (list[int | float]): Input sequence.
        nd (int): Decimal places. Defaults to 3.

    Returns:
        list[float]: Rounded float list.
    """
    return [round(float(x), nd) for x in xs]


def slope(a: Union[List[float], np.ndarray]) -> float:
    """
    Compute the slope (trend) of a numeric sequence via linear regression.

    Args:
        a (list | np.ndarray): Input numeric sequence.

    Returns:
        float: Slope of best-fit line; 0.0 if sequence too short.
    """
    a = np.array(a, dtype=float)
    return 0.0 if len(a) < 3 else round(np.polyfit(np.arange(len(a)), a, 1)[0], 4)


def z(a: Union[List[float], np.ndarray]) -> List[float]:
    """
    Compute the z-score normalization of a numeric sequence.

    Args:
        a (list | np.ndarray): Input numeric sequence.

    Returns:
        list[float]: Z-scored values (mean 0, std 1). Returns zeros if std=0 or len<3.
    """
    a = np.array(a, dtype=float)
    return [0.0] * len(a) if len(a) < 3 or np.std(a) == 0 else (
        ((a - a.mean()) / a.std()).round(3).tolist()
    )
