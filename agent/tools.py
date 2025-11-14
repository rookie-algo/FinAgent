import json

from langchain.tools import tool
import numpy as np
import pandas as pd
import talib
from talib import MA_Type

from agent.utils import get_price_history, last, minmax01, \
      tail, to_float_list, slope, z


@tool("trend_detection")
def trend_detection(symbol: str) -> str:
    """
    Extract numeric trend-related features from OHLCV data.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    # Indicators
    ema_5  = talib.EMA(close, 5)
    ema_12 = talib.EMA(close, 12)
    ema_26 = talib.EMA(close, 26)
    ema12_26 = ema_12 - ema_26

    macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
    adx      = talib.ADX(high, low, close, 14)
    plus_di  = talib.PLUS_DI(high, low, close, 14)
    minus_di = talib.MINUS_DI(high, low, close, 14)
    cci      = talib.CCI(high, low, close, 14)

    # assemble lists
    lists = {
        "ema_5": tail(ema_5, k),
        "ema_12": tail(ema_12, k),
        "ema_26": tail(ema_26, k),
        "ema12_minus_ema26": tail(ema12_26, k),
        "macd": tail(macd, k),
        "macd_signal": tail(macd_sig, k),
        "macd_hist": tail(macd_hist, k),
        "adx": tail(adx, k),
        "plus_di": tail(plus_di, k),
        "minus_di": tail(minus_di, k),
        "cci": tail(cci, k),
    }

    for key, v in lists.items():
        if len(v) < k:
            return json.dumps({"status": "insufficient_data", "reason": f"{key}_len<{k}"})

    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    features = {
        "ema_spread_slope": float(slope(lists["ema12_minus_ema26"])),
        "macd_hist_slope":  float(slope(lists["macd_hist"])),
        "adx_slope":        float(slope(lists["adx"])),
        "cci_z":            to_float_list(z(lists["cci"]), 3),
    }

    return {
        "tool": "trend_detection",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": features,
    }


@tool("momentum_strength")
def momentum_strength(symbol: str) -> dict:
    """
    Extract numeric momentum-related features from OHLCV data.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Indicators
    rsi = tail(talib.RSI(close, 14), k)
    slowk, slowd = talib.STOCH(high, low, close)
    slowk, slowd = tail(slowk, k), tail(slowd, k)
    roc = tail(talib.ROC(close), k)
    mfi = tail(talib.MFI(high, low, close, vol), k)

    # Required lists
    lists = {
        "rsi": rsi,
        "slowk": slowk,
        "slowd": slowd,
        "roc": roc,
        "mfi": mfi,
    }

    # Validate
    for key, v in lists.items():
        if len(v) < k:
            return {
                "tool": "momentum_strength",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": f"{key}_len<{k}"
            }

    # Convert to lists of floats (3 decimals)
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helper features
    features = {
        "rsi_slope": float(slope(lists["rsi"])),
        "stoch_diff_slope": float(slope(np.array(lists["slowk"]) - np.array(lists["slowd"]))),
        "roc_slope": float(slope(lists["roc"])),
        "mfi_slope": float(slope(lists["mfi"])),
        "rsi_z": to_float_list(z(lists["rsi"]), 3),
        "mfi_z": to_float_list(z(lists["mfi"]), 3),
    }

    return {
        "tool": "momentum_strength",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": features,
    }


@tool("volatility_range")
def volatility_range(symbol: str) -> dict:
    """
    Extract numeric volatility and range-based features from OHLCV data.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    # Indicators
    atr = tail(talib.ATR(high, low, close, 14), k)
    upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
    upper, middle, lower = tail(upper, k), tail(middle, k), tail(lower, k)
    stdv5 = tail(talib.STDDEV(close, timeperiod=5, nbdev=1), k)

    # Required lists
    lists = {
        "atr": atr,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "stddev_5": stdv5,
    }

    # Validate
    for key, v in lists.items():
        if len(v) < k:
            return {
                "tool": "volatility_range",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": f"{key}_len<{k}",
            }

    # Cast lists → floats
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helper features
    bw = np.array(lists["bb_upper"]) - np.array(lists["bb_lower"])
    recent_close = close.dropna().tail(k).values.astype(float)
    denom = np.maximum(bw, 1e-12)

    pos = np.clip((recent_close - np.array(lists["bb_lower"])) / denom, 0.0, 1.0)

    helpers = {
        "band_width": to_float_list(bw, 3),
        "band_width_slope": float(slope(bw)),
        "band_position": to_float_list(pos, 3),
        "atr_slope": float(slope(lists["atr"])),
        "stddev_slope": float(slope(lists["stddev_5"])),
        "atr_z": to_float_list(z(lists["atr"]), 3),
        "stddev_z": to_float_list(z(lists["stddev_5"]), 3),
    }

    return {
        "tool": "volatility_range",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": helpers,
    }


@tool("volume_flow")
def volume_flow(symbol: str) -> dict:
    """
    Extract numeric volume-flow and participation features from OHLCV data.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Raw indicators
    obv = tail(talib.OBV(close, vol), k)
    ad  = tail(talib.AD(high, low, close, vol), k)
    v10 = tail(talib.EMA(vol, 10), k)
    v20 = tail(talib.EMA(vol, 20), k)

    # Required lists
    lists = {
        "obv": obv,
        "ad": ad,
        "volume_ema_10": v10,
        "volume_ema_20": v20,
    }

    # Validate length
    for key, v in lists.items():
        if len(v) < k:
            return {
                "tool": "volume_flow",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": f"{key}_len<{k}",
            }

    # Normalize to floats
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helpers
    helpers = {
        "obv_slope": float(slope(lists["obv"])),
        "ad_slope": float(slope(lists["ad"])),
        "volume_ratio": round(
            lists["volume_ema_10"][-1] / (lists["volume_ema_20"][-1] + 1e-12),
            3,
        ),
        "obv_z": to_float_list(z(lists["obv"]), 3),
        "ad_z": to_float_list(z(lists["ad"]), 3),
    }

    return {
        "tool": "volume_flow",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": helpers,
    }


@tool("market_structure")
def market_structure(symbol: str) -> dict:
    """
    Extract numeric market-structure features from OHLCV data.
    Includes swings, gaps, candle ratios, and range statistics.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    df = get_price_history(symbol=symbol)
    k = 14

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    # --- Swings ---
    swing_high_mask = (h.shift(1) < h) & (h.shift(-1) < h)
    swing_low_mask  = (l.shift(1) > l) & (l.shift(-1) > l)
    swing_high_vals = h.where(swing_high_mask)
    swing_low_vals  = l.where(swing_low_mask)

    # --- Range + Position ---
    recent_high = h.rolling(k, min_periods=1).max()
    recent_low  = l.rolling(k, min_periods=1).min()
    rng = recent_high - recent_low
    price_pos = np.clip(
        (c - recent_low) / (rng.replace(0, np.nan)).fillna(1e-12), 0, 1
    )

    # --- Candle shape ---
    tr = (h - l).replace(0, 1e-12)
    body = (c - o).abs()
    upper = (h - np.maximum(o, c))
    lower = (np.minimum(o, c) - l)
    body_ratio = (body / tr).clip(0, 1)
    upper_wick = (upper / tr).clip(0, 1)
    lower_wick = (lower / tr).clip(0, 1)

    # --- Gaps ---
    prev_close = c.shift(1)
    gap = o - prev_close
    gap_up = (gap > 0).astype(int)
    gap_down = (gap < 0).astype(int)

    # --- Efficiency ---
    net_change = (c - c.shift(k)).abs()
    path = c.diff().abs().rolling(k).sum().replace(0, np.nan)
    efficiency = (net_change / path).fillna(0).clip(0, 1)

    # --- Support / Resistance Clustering ---
    def _cluster(vals, n=3):
        x = vals.dropna().tail(100).values.astype(float)
        if x.size == 0:
            return []
        hist, edges = np.histogram(x, bins=20)
        idx = hist.argsort()[::-1][:n]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in idx]
        return sorted([round(float(v), 3) for v in centers])

    supports = _cluster(swing_low_vals)
    resistances = _cluster(swing_high_vals)

    # --- Lists (last k) ---
    lists = {
        "highs":          tail(h, k),
        "lows":           tail(l, k),
        "closes":         tail(c, k),
        "range_width":    tail(rng, k),
        "price_position": tail(price_pos, k),
        "body_ratio":     tail(body_ratio, k),
        "upper_wick_ratio": tail(upper_wick, k),
        "lower_wick_ratio": tail(lower_wick, k),
        "gap_up":         tail(gap_up, k),
        "gap_down":       tail(gap_down, k),
        "swing_highs":    tail(swing_high_vals, k),
        "swing_lows":     tail(swing_low_vals, k),
        "efficiency":     tail(efficiency, k),
    }

    # Validate list lengths
    for key, arr in lists.items():
        if len(arr) < k:
            return {
                "tool": "market_structure",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": f"{key}_len<{k}",
            }

    # JSON-safe lists
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # --- Helper features ---
    helpers = {
        "range_width_slope": slope(lists["range_width"]),
        "price_position_last": float(lists["price_position"][-1]),
        "efficiency_last": float(lists["efficiency"][-1]),
        "gap_rate_up": float(np.mean(lists["gap_up"])),
        "gap_rate_down": float(np.mean(lists["gap_down"])),
        "swing_high_count": int(np.isfinite(np.array(lists["swing_highs"])).sum()),
        "swing_low_count": int(np.isfinite(np.array(lists["swing_lows"])).sum()),
        "support_levels": supports,
        "resistance_levels": resistances,
        "range_width_z": to_float_list(z(lists["range_width"]), 3),
        "body_ratio_z": to_float_list(z(lists["body_ratio"]), 3),
    }

    return {
        "tool": "market_structure",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": helpers,
    }


@tool("liquidity_participation")
def liquidity_participation(symbol: str) -> dict:
    """
    Extract numeric liquidity and participation features from OHLCV + microstructure fields.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    df = get_price_history(symbol=symbol)

    # Required OHLCV
    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    vol   = pd.to_numeric(df["Volume"], errors="coerce")

    # Optional microstructure fields
    flt    = pd.to_numeric(df.get("Float", pd.Series(index=df.index)), errors="coerce")
    trades = pd.to_numeric(df.get("Trades", pd.Series(index=df.index)), errors="coerce")
    tickv  = pd.to_numeric(df.get("TickVolume", pd.Series(index=df.index)), errors="coerce")
    bid    = pd.to_numeric(df.get("Bid", pd.Series(index=df.index)), errors="coerce")
    ask    = pd.to_numeric(df.get("Ask", pd.Series(index=df.index)), errors="coerce")

    # Core metric: Dollar Volume
    dollar_volume = close * vol

    # Spread proxy: prefer bid/ask
    if bid.notna().any() and ask.notna().any():
        mid = ((bid + ask) / 2).replace(0, np.nan)
        spread_proxy = ((ask - bid) / mid).replace([np.inf, -np.inf], np.nan)
    else:
        denom = np.maximum(close.replace(0, np.nan), 1e-12)
        spread_proxy = ((high - low) / denom).replace([np.inf, -np.inf], np.nan)

    # Volume volatility
    vol_sigma = vol.rolling(20, min_periods=5).std()
    vol_mu    = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    volume_volatility = (vol_sigma / vol_mu).replace([np.inf, -np.inf], np.nan)

    # Optional turnover ratio
    turnover_ratio = None
    if flt.notna().sum() > 0:
        denom = flt.replace(0, np.nan)
        turnover_ratio = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # Optional avg trade size
    avg_trade_size = None
    if trades.notna().sum() > 0:
        denom = trades.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)
    elif tickv.notna().sum() > 0:
        denom = tickv.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # Assemble lists
    lists = {
        "dollar_volume":     tail(dollar_volume, k),
        "spread_proxy":      tail(spread_proxy, k),
        "volume_volatility": tail(volume_volatility, k),
    }
    if turnover_ratio is not None:
        lists["turnover_ratio"] = tail(turnover_ratio, k)
    if avg_trade_size is not None:
        lists["avg_trade_size"] = tail(avg_trade_size, k)

    # Validate core lists
    for ck in ["dollar_volume", "spread_proxy", "volume_volatility"]:
        if len(lists.get(ck, [])) < k:
            return {
                "tool": "liquidity_participation",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": "core_features_len<14",
            }

    # Convert to JSON-safe floats
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helpers
    helpers = {
        "dv_slope": slope(lists["dollar_volume"]),
        "spread_slope": slope(lists["spread_proxy"]),
        "volvol_slope": slope(lists["volume_volatility"]),
        "dv_z": to_float_list(z(lists["dollar_volume"]), 3),
        "spread_z": to_float_list(z(lists["spread_proxy"]), 3),
        "volvol_z": to_float_list(z(lists["volume_volatility"]), 3),
        "turnover_slope": slope(lists["turnover_ratio"]) if "turnover_ratio" in lists else None,
        "avg_trade_size_slope": slope(lists["avg_trade_size"]) if "avg_trade_size" in lists else None,
        "has_turnover": int("turnover_ratio" in lists),
        "has_avg_trade_size": int("avg_trade_size" in lists),
    }

    return {
        "tool": "liquidity_participation",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": helpers,
    }


@tool("risk_efficiency")
def risk_efficiency(symbol: str) -> dict:
    """
    Extract numeric risk and efficiency features from OHLCV data.
    Includes Sharpe/Sortino proxies, drawdown, volatility and path efficiency.
    Returns raw values and helper signals only.
    No prompts, no schemas, no instructions.
    Safe for multi-tool execution.
    """
    k = 14
    W = 20
    ANNUAL = np.sqrt(252)

    df = get_price_history(symbol)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)

    # Sanity: need enough history
    if close.dropna().shape[0] < (W + k):
        return {
            "tool": "risk_efficiency",
            "symbol": symbol,
            "status": "insufficient_data",
            "reason": "not_enough_prices",
        }

    # Returns
    ret = close.pct_change()

    # Rolling statistics
    mu = ret.rolling(W, min_periods=W).mean()
    sd = ret.rolling(W, min_periods=W).std()
    dsd = ret.clip(upper=0).rolling(W, min_periods=W).std()

    sharpe = (mu / sd.replace(0, np.nan)) * ANNUAL
    sortino = (mu / dsd.replace(0, np.nan)) * ANNUAL

    wealth = (1 + ret.fillna(0)).cumprod()
    roll_max = wealth.rolling(W, min_periods=W).max()
    dd = wealth / roll_max.replace(0, np.nan) - 1
    mdd = dd.rolling(W, min_periods=W).min().abs()
    ulcer = np.sqrt(dd.pow(2).rolling(W, min_periods=W).mean())

    delta = close.diff()
    path_len = delta.abs().rolling(W, min_periods=W).sum()
    net_move = (close - close.shift(W)).abs()
    er = (net_move / path_len.replace(0, np.nan)).clip(0, 1)

    # VaR & ES
    def _var_es_95(x):
        x = pd.Series(x).dropna()
        if x.shape[0] < W:
            return np.nan, np.nan
        q = np.quantile(x, 0.05)
        es = x[x <= q].mean() if (x <= q).any() else q
        return float(-q), float(-es)

    var_95 = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(s)[0])
    es_95  = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(s)[1])

    # Collect last 14 points
    lists = {
        "sharpe_rolling":     tail(sharpe, k),
        "sortino_rolling":    tail(sortino, k),
        "mdd_rolling":        tail(mdd, k),
        "ulcer_index":        tail(ulcer, k),
        "eff_ratio":          tail(er, k),
        "var_95":             tail(var_95, k),
        "es_95":              tail(es_95, k),
    }

    # Validate
    for key, arr in lists.items():
        if len(arr) < k:
            return {
                "tool": "risk_efficiency",
                "symbol": symbol,
                "status": "insufficient_data",
                "reason": f"{key}_len<{k}",
            }

    # Precision + JSON-safe
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    helpers = {
        "sharpe_slope": slope(lists["sharpe_rolling"]),
        "eff_ratio_slope": slope(lists["eff_ratio"]),
        "mdd_slope": slope(lists["mdd_rolling"]),
        "ulcer_slope": slope(lists["ulcer_index"]),
        "var95_last": lists["var_95"][-1],
        "es95_last": lists["es_95"][-1],
        "eff_ratio_last": lists["eff_ratio"][-1],
        "sharpe_z": to_float_list(z(lists["sharpe_rolling"]), 3),
        "ulcer_z": to_float_list(z(lists["ulcer_index"]), 3),
    }

    return {
        "tool": "risk_efficiency",
        "symbol": symbol,
        "status": "ok",
        "lookback_points": k,
        "data": lists,
        "features": helpers,
    }


TOOLS = [trend_detection, momentum_strength, volatility_range, volume_flow,
         market_structure, liquidity_participation, risk_efficiency]
