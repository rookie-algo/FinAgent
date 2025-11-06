import numpy as np
import yfinance as yf


def _fetch_finance_df_with_symbol(symbol: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='1y', interval='1d')
    return df


def _last(arr): return float(arr[-1]) if len(arr) else 0.0


def _minmax01(x):
    arr = np.array(x, float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx-mn == 0: 
        return np.zeros_like(arr, dtype=float).tolist()
    return ((arr - mn) / (mx - mn)).clip(0,1).tolist()


def _tail(s, k=14, d=3):
    return s.dropna().tail(k).round(d).tolist()


def _to_float_list(xs, nd=3):
    return [round(float(x), nd) for x in xs]


def _slope(a):
    a = np.array(a, float)
    return 0.0 if len(a) < 3 else round(np.polyfit(np.arange(len(a)), a, 1)[0], 4)


def _z(a):
    a = np.array(a, float)
    return [0.0]*len(a) if len(a) < 3 or np.std(a) == 0 else ((a - a.mean())/a.std()).round(3).tolist()
