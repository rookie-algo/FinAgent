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
    Generate a STRICT-JSON trend-analysis prompt using cached OHLCV data.
    Returns a compact instruction prompt and 14-point numeric features.
    No advice. No indicator names in explanation/rationale.
    Output must follow the exact JSON schema included in the prompt.
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

    data_json    = json.dumps(lists, separators=(",", ":"))
    helpers_json = json.dumps(features, separators=(",", ":"))
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k})

    # --------- Compact Prompt (TPM-SAFE) ---------
    prompt = f"""
STRICT JSON ONLY. No markdown. No extra keys.
If data insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "direction":"up"|"down"|"sideways"|"mixed",
    "strength":"weak"|"moderate"|"strong",
    "momentum":"building"|"fading"|"stable",
    "confidence":0..1,
    "explanation":"Plain-language summary. No indicator names.",
    "rationale":"Concise numeric justification. No indicator names."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric data below.
- Base direction/strength/momentum on consistency + magnitude.
- Confidence reflects numeric agreement (never 1.0).
- No advice. No predictions.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("momentum_strength")
def momentum_strength(symbol: str) -> str:
    """
    Return a compact STRICT-JSON momentum-analysis prompt using cached OHLCV data.
    Output summarizes momentum_state, strength, pressure, and confidence.
    No advice. No indicator names. Returns only an instruction prompt + features.
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

    for key, v in lists.items():
        if len(v) < k:
            return json.dumps({"status": "insufficient_data", "reason": f"{key}_len<{k}"})

    # JSON-safe floats
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

    data_json = json.dumps(lists, separators=(",", ":"))
    features_json = json.dumps(features, separators=(",", ":"))
    header_json = json.dumps({"symbol": symbol, "lookback_points": k})

    prompt = f"""
STRICT JSON ONLY. No markdown, no extra keys.
If data insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "momentum_state":"rising"|"falling"|"neutral"|"mixed",
    "strength":"weak"|"moderate"|"strong",
    "pressure":"buying"|"selling"|"balanced",
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- Momentum up/down based on consistent direction.
- Strength = magnitude + agreement.
- Pressure = buyer vs seller dominance.
- Confidence < 1.0, based on numeric agreement.
- No advice, predictions, or targets.

DATA:
{data_json}

FEATURES:
{features_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("volatility_range")
def volatility_range(symbol: str) -> str:
    """
    Return a compact STRICT-JSON volatility-analysis prompt using cached OHLCV.
    Summarizes volatility_regime, volatility_level, and relative_position.
    No advice. No indicator names. Returns instruction prompt + numeric features.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    # Indicators
    atr = tail(talib.ATR(high, low, close, 14), k)
    upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
    upper, middle, lower = tail(upper, k), tail(middle, k), tail(lower, k)
    stdv5 = tail(talib.STDDEV(close, timeperiod=5, nbdev=1), k)

    # Required series
    lists = {
        "atr": atr,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "stddev_5": stdv5,
    }

    for key, v in lists.items():
        if len(v) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    # JSON-safe floats
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helper features
    bw = np.array(lists["bb_upper"]) - np.array(lists["bb_lower"])  # band width
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

    data_json = json.dumps(lists, separators=(",", ":"))
    helpers_json = json.dumps(helpers, separators=(",", ":"))
    header_json = json.dumps({"symbol": symbol, "lookback_points": k})

    # ============ TPM SAFE PROMPT (compact) ============
    prompt = f"""
STRICT JSON ONLY. No markdown, no extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "volatility_regime":"expanding"|"contracting"|"stable"|"mixed",
    "volatility_level":"low"|"medium"|"high",
    "relative_position":"near_high"|"near_low"|"mid_range",
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- Regime: dispersion ↑ = expanding; ↓ = contracting.
- Level: recent magnitude vs own history.
- Position: 0=lower band, 1=upper.
- Confidence <1.0, based on numeric agreement.
- No advice, predictions, or targets.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("volume_flow")
def volume_flow(symbol: str) -> str:
    """
    Return a compact STRICT-JSON volume/flow analysis prompt from OHLCV.
    Summarizes volume_trend, flow_bias, participation_strength, and confidence.
    No advice. No indicator names.
    """
    k = 14
    df = get_price_history(symbol=symbol)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Indicators
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
    for key, v in lists.items():
        if len(v) < k:
            return json.dumps({"status": "insufficient_data", "reason": f"{key}_len<{k}"})

    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    # Helper features
    features = {
        "obv_slope": float(slope(lists["obv"])),
        "ad_slope": float(slope(lists["ad"])),
        "volume_ratio": round(lists["volume_ema_10"][-1] / (lists["volume_ema_20"][-1] + 1e-12), 3),
        "obv_z": to_float_list(z(lists["obv"]), 3),
        "ad_z": to_float_list(z(lists["ad"]), 3),
    }

    data_json = json.dumps(lists, separators=(",", ":"))
    features_json = json.dumps(features, separators=(",", ":"))
    header_json = json.dumps({"symbol": symbol, "lookback_points": k})

    # ============= TPM-SAFE COMPACT PROMPT =================
    prompt = f"""
STRICT JSON ONLY. No markdown, no extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "volume_trend":"increasing"|"decreasing"|"stable"|"mixed",
    "flow_bias":"inflow"|"outflow"|"balanced",
    "participation_strength":"weak"|"moderate"|"strong",
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- Volume trend: compare short vs long activity and slopes.
- Flow bias: agree between obv/ad direction (inflow/outflow).
- Participation strength: magnitude + consistency.
- Confidence < 1.0, based on agreement.
- No advice, predictions, or targets.

DATA:
{data_json}

FEATURES:
{features_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("market_structure")
def market_structure(symbol: str) -> str:
    """
    Return a compact STRICT-JSON market-structure prompt from OHLCV.
    Summarizes structure, bias, price_position, key_levels, and confidence.
    No advice. No indicator names.
    """
    df = get_price_history(symbol=symbol)
    k = 14

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    # --- Swings (window=3)
    swing_high_mask = (h.shift(1) < h) & (h.shift(-1) < h)
    swing_low_mask  = (l.shift(1) > l) & (l.shift(-1) > l)
    swing_high_vals = h.where(swing_high_mask)
    swing_low_vals  = l.where(swing_low_mask)

    # --- Range + Position
    recent_high = h.rolling(k, min_periods=1).max()
    recent_low  = l.rolling(k, min_periods=1).min()
    rng = recent_high - recent_low
    price_pos = np.clip((c - recent_low) / (rng.replace(0, np.nan)).fillna(1e-12), 0, 1)

    # --- Candle shape
    tr = (h - l).replace(0, 1e-12)
    body = (c - o).abs()
    upper = (h - np.maximum(o, c))
    lower = (np.minimum(o, c) - l)
    body_ratio = (body / tr).clip(0, 1)
    uw = (upper / tr).clip(0, 1)
    lw = (lower / tr).clip(0, 1)

    # --- Gaps
    prev_close = c.shift(1)
    gap_up = ((o - prev_close) > 0).astype(int)
    gap_down = ((o - prev_close) < 0).astype(int)

    # --- Efficiency
    net_change = (c - c.shift(k)).abs()
    path = c.diff().abs().rolling(k).sum().replace(0, np.nan)
    efficiency = (net_change / path).fillna(0).clip(0, 1)

    # --- Simple support/resistance clustering
    def _cluster(vals, n=3):
        x = vals.dropna().tail(100).values.astype(float)
        if x.size == 0:
            return []
        hist, edges = np.histogram(x, bins=20)
        idx = hist.argsort()[::-1][:n]
        centers = [(edges[i] + edges[i+1]) / 2 for i in idx]
        return sorted([round(float(v), 3) for v in centers])

    supports = _cluster(swing_low_vals)
    resistances = _cluster(swing_high_vals)

    # --- Lists (last k points)
    lists = {
        "highs":          tail(h, k),
        "lows":           tail(l, k),
        "closes":         tail(c, k),
        "range_width":    tail(rng, k),
        "price_position": tail(price_pos, k),
        "body_ratio":     tail(body_ratio, k),
        "upper_wick_ratio": tail(uw, k),
        "lower_wick_ratio": tail(lw, k),
        "gap_up":         tail(gap_up, k),
        "gap_down":       tail(gap_down, k),
        "swing_highs":    tail(swing_high_vals, k),
        "swing_lows":     tail(swing_low_vals, k),
        "efficiency":     tail(efficiency, k),
    }

    for key, arr in lists.items():
        if len(arr) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    # --- Helper features
    helpers = {
        "range_width_slope": slope(lists["range_width"]),
        "price_position_last": float(lists["price_position"][-1]),
        "efficiency_last": float(lists["efficiency"][-1]),
        "gap_rate_up": float(np.mean(lists["gap_up"])),
        "gap_rate_down": float(np.mean(lists["gap_down"])),
        "swing_high_count": int(pd.Series(lists["swing_highs"]).count()),
        "swing_low_count": int(pd.Series(lists["swing_lows"]).count()),
        "support_levels": supports,
        "resistance_levels": resistances,
        "range_width_z": to_float_list(z(lists["range_width"]), 3),
        "body_ratio_z": to_float_list(z(lists["body_ratio"]), 3),
    }

    data_json    = json.dumps({k: to_float_list(v, 3) for k, v in lists.items()},
                              separators=(",", ":"))
    helpers_json = json.dumps(helpers, separators=(",", ":"))
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k})

    prompt = f"""
STRICT JSON ONLY. No markdown, no extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "structure":"trend"|"range"|"transition"|"mixed",
    "bias":"up"|"down"|"neutral"|"mixed",
    "price_position":"near_high"|"near_low"|"mid_range",
    "key_levels":{{"support":[number,...],"resistance":[number,...]}},
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- Structure: efficiency + swing patterns + range width direction.
- Bias: higher highs vs lower lows, drift in price_position.
- Price_position: last value → near_low (<0.33), mid_range (0.33–0.66), near_high (>=0.66).
- Key levels: use provided support/resistance arrays.
- Confidence < 1.0, based on agreement.
- No advice, predictions, or targets.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("liquidity_participation")
def liquidity_participation(symbol: str) -> str:
    """
    Compact STRICT-JSON liquidity/participation analysis prompt from OHLCV + optional microstructure data.
    Outputs liquidity_level, spread_state, participation_quality, and confidence. No advice. No indicator names.
    """
    k = 14
    df = get_price_history(symbol=symbol)

    # Required OHLCV
    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    vol   = pd.to_numeric(df["Volume"], errors="coerce")

    # Optional microstructure inputs
    flt    = pd.to_numeric(df.get("Float", pd.Series(index=df.index)), errors="coerce")
    trades = pd.to_numeric(df.get("Trades", pd.Series(index=df.index)), errors="coerce")
    tickv  = pd.to_numeric(df.get("TickVolume", pd.Series(index=df.index)), errors="coerce")
    bid    = pd.to_numeric(df.get("Bid", pd.Series(index=df.index)), errors="coerce")
    ask    = pd.to_numeric(df.get("Ask", pd.Series(index=df.index)), errors="coerce")

    # --- Core metrics ---
    dollar_volume = close * vol

    # Spread: prefer Bid/Ask if present
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

    # Optional: turnover ratio
    turnover_ratio = None
    if flt.notna().sum() > 0:
        denom = flt.replace(0, np.nan)
        turnover_ratio = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # Optional: avg trade size
    avg_trade_size = None
    if trades.notna().sum() > 0:
        denom = trades.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)
    elif tickv.notna().sum() > 0:
        denom = tickv.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # --- Assemble lists ---
    lists = {
        "dollar_volume":     tail(dollar_volume, k),
        "spread_proxy":      tail(spread_proxy, k),
        "volume_volatility": tail(volume_volatility, k),
    }
    if turnover_ratio is not None:
        lists["turnover_ratio"] = tail(turnover_ratio, k)
    if avg_trade_size is not None:
        lists["avg_trade_size"] = tail(avg_trade_size, k)

    # Required core lists must have 14 points
    for ck in ["dollar_volume", "spread_proxy", "volume_volatility"]:
        if len(lists.get(ck, [])) < k:
            return json.dumps({"status": "insufficient_data", "reason": "core_features_len<14"})

    # JSON-safe floats
    lists = {k: to_float_list(v, 3) for k, v in lists.items()}

    # --- Helper features ---
    helpers = {
        "dv_slope": float(slope(lists["dollar_volume"])),
        "spread_slope": float(slope(lists["spread_proxy"])),
        "volvol_slope": float(slope(lists["volume_volatility"])),
        "dv_z": to_float_list(z(lists["dollar_volume"]), 3),
        "spread_z": to_float_list(z(lists["spread_proxy"]), 3),
        "volvol_z": to_float_list(z(lists["volume_volatility"]), 3),
        "turnover_slope": float(slope(lists["turnover_ratio"])) if "turnover_ratio" in lists else None,
        "avg_trade_size_slope": float(slope(lists["avg_trade_size"])) if "avg_trade_size" in lists else None,
        "has_turnover": int("turnover_ratio" in lists),
        "has_avg_trade_size": int("avg_trade_size" in lists),
    }

    data_json    = json.dumps(lists, separators=(",", ":"))
    helpers_json = json.dumps(helpers, separators=(",", ":"))
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k})
    prompt = f"""
STRICT JSON ONLY. No markdown, no extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "liquidity_level":"low"|"medium"|"high",
    "spread_state":"tight"|"normal"|"wide",
    "participation_quality":"institutional"|"retail"|"mixed"|"insufficient",
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- Liquidity level: based on dollar_volume level/slope and turnover_ratio if present.
- Spread_state: magnitude + direction of spread_proxy.
- Participation_quality: use avg_trade_size consistency/level; if absent → "insufficient".
- Confidence <1.0 and based on agreement.
- No predictions or advice.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("risk_efficiency")
def risk_efficiency(symbol: str) -> str:
    """
    Compact STRICT-JSON prompt for risk/efficiency analysis from OHLCV.
    Computes rolling Sharpe, Sortino, MDD, Ulcer, efficiency ratio, VaR, ES.
    Outputs risk_regime, efficiency_state, downside_risk, confidence.
    No advice. No indicator names.
    """
    k = 14
    W = 20
    ANNUAL = np.sqrt(252)

    df = get_price_history(symbol)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)

    # Need at least W+k prices
    if close.dropna().shape[0] < W + k:
        return json.dumps({"status": "insufficient_data", "reason": "not_enough_prices"})

    # Returns
    ret = close.pct_change()

    # Rolling mean/std & downside std
    mu = ret.rolling(W, min_periods=W).mean()
    sd = ret.rolling(W, min_periods=W).std()
    dsd = ret.clip(upper=0).rolling(W, min_periods=W).std()

    # Sharpe/Sortino
    sharpe = (mu / sd.replace(0, np.nan)) * ANNUAL
    sortino = (mu / dsd.replace(0, np.nan)) * ANNUAL

    # Wealth → Drawdown → MDD & Ulcer
    wealth = (1 + ret.fillna(0)).cumprod()
    roll_max = wealth.rolling(W, min_periods=W).max()
    dd = wealth / roll_max.replace(0, np.nan) - 1  # <=0
    mdd = dd.rolling(W, min_periods=W).min().abs()
    ulcer = np.sqrt(dd.pow(2).rolling(W, min_periods=W).mean())

    # Efficiency Ratio
    delta = close.diff()
    path_len = delta.abs().rolling(W, min_periods=W).sum()
    net_move = (close - close.shift(W)).abs()
    er = (net_move / path_len.replace(0, np.nan)).clip(0, 1)

    # VaR95 & ES95
    def _var_es_95(x):
        x = pd.Series(x).dropna()
        if x.shape[0] < W:
            return np.nan, np.nan
        q = np.quantile(x, 0.05)
        es = x[x <= q].mean() if (x <= q).any() else q
        return float(-q), float(-es)

    var_95 = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(s)[0])
    es_95  = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(s)[1])

    # Assemble last 14 entries
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
    for k_, v in lists.items():
        if len(v) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{k_}_len<{k}"})

    # Float & precision
    lists = {k_: to_float_list(v, 3) for k_, v in lists.items()}

    helpers = {
        "sharpe_slope": float(slope(lists["sharpe_rolling"])),
        "eff_ratio_slope": float(slope(lists["eff_ratio"])),
        "mdd_slope": float(slope(lists["mdd_rolling"])),
        "ulcer_slope": float(slope(lists["ulcer_index"])),
        "var95_last": float(lists["var_95"][-1]),
        "es95_last": float(lists["es_95"][-1]),
        "eff_ratio_last": float(lists["eff_ratio"][-1]),
        "sharpe_z": to_float_list(z(lists["sharpe_rolling"]), 3),
        "ulcer_z": to_float_list(z(lists["ulcer_index"]), 3),
    }

    data_json    = json.dumps(lists, separators=(",", ":"))
    helpers_json = json.dumps(helpers, separators=(",", ":"))
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k})

    prompt = f"""
STRICT JSON ONLY. No markdown. No extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

Schema:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "risk_regime":"muted"|"normal"|"elevated"|"mixed",
    "efficiency_state":"efficient"|"noisy"|"mixed",
    "downside_risk":"low"|"medium"|"high",
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

Rules:
- Use ONLY numeric values below.
- risk_regime: based on var_95, es_95, ulcer_index, mdd trend/level.
- efficiency_state: based on eff_ratio level/slope.
- downside_risk: reflect mdd/es95/ulcer recency.
- confidence <1.0, based on agreement.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt


@tool("regime_composite")
def regime_composite(symbol: str) -> str:
    """
    Compact STRICT-JSON regime/composite scoring prompt.
    Computes 0–100 trend/momentum/volatility/flow scores, composite, regimes, transitions.
    No advice. No indicator names.
    """

    k = 14
    W = 20  # internal rolling window
    weights = {"trend": 0.3, "momentum": 0.3, "volatility": 0.2, "flow": 0.2}

    df = get_price_history(symbol=symbol)
    o = pd.to_numeric(df["Open"], errors="coerce").astype(float)
    h = pd.to_numeric(df["High"], errors="coerce").astype(float)
    l = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    v = pd.to_numeric(df["Volume"], errors="coerce").astype(float)

    # ---------- base series ----------
    ema12 = talib.EMA(c, 12)
    ema26 = talib.EMA(c, 26)
    spread = ema12 - ema26
    macd, macd_sig, macd_hist = talib.MACD(c, 12, 26, 9)
    adx = talib.ADX(h, l, c, 14)
    rsi = talib.RSI(c, 14)
    slowk, slowd = talib.STOCH(h, l, c)
    roc = talib.ROC(c)
    atr = talib.ATR(h, l, c, 14)
    bb_u, bb_m, bb_l = talib.BBANDS(c, matype=MA_Type.T3)
    obv = talib.OBV(c, v)
    ad  = talib.AD(h, l, c, v)
    vol_ema10 = talib.EMA(v, 10)
    vol_ema20 = talib.EMA(v, 20)

    # efficiency (price path directionality)
    net_change = (c - c.shift(W)).abs()
    path = c.diff().abs().rolling(W).sum()
    eff = (net_change / path.replace(0, np.nan)).fillna(0.0).clip(0,1)

    # band width
    bw = (bb_u - bb_l)

    # ---------- component scoring (0–100) over full series, then emit last 14) ----------
    # Trend: positive MACD hist slope + ADX level + spread slope → higher
    trend_raw = (
        minmax01(macd_hist) + 
        minmax01(adx) + 
        minmax01(spread)
    )
    trend_score_full = np.round(100 * np.array(minmax01(pd.Series(trend_raw).rolling(W).mean())), 3).tolist()

    # Momentum: RSI z (abs/level), ROC slope, STOCH diff slope
    stoch_diff = (slowk - slowd)
    mom_raw = (
        minmax01(rsi) + 
        minmax01(roc) + 
        minmax01(stoch_diff)
    )
    momentum_score_full = np.round(100 * np.array(minmax01(pd.Series(mom_raw).rolling(W).mean())), 3).tolist()

    # Volatility: ATR z + BW slope (higher ⇒ more volatile)
    vol_raw = (
        minmax01(atr) + 
        minmax01(bw)
    )
    volatility_score_full = np.round(100 * np.array(minmax01(pd.Series(vol_raw).rolling(W).mean())), 3).tolist()

    # Flow: OBV/AD slope + volume ratio (10/20)
    vol_ratio = (vol_ema10 / (vol_ema20.replace(0, np.nan))).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    flow_raw = (
        minmax01(obv) + 
        minmax01(ad) + 
        minmax01(vol_ratio)
    )
    flow_score_full = np.round(100 * np.array(minmax01(pd.Series(flow_raw).rolling(W).mean())), 3).tolist()

    # Synchronized tail (last 14)
    trend_score     = tail(pd.Series(trend_score_full))
    momentum_score  = tail(pd.Series(momentum_score_full))
    volatility_score= tail(pd.Series(volatility_score_full))
    flow_score      = tail(pd.Series(flow_score_full))

    # Sanity: all components need 14 values
    for kx in (trend_score, momentum_score, volatility_score, flow_score):
        if len(kx) < k:
            return json.dumps({"status":"insufficient_data","reason":"component_scores_len<14"})

    # Composite (weights)
    comp = (
        np.array(trend_score)*weights["trend"] +
        np.array(momentum_score)*weights["momentum"] +
        np.array(volatility_score)*weights["volatility"] +
        np.array(flow_score)*weights["flow"]
    )
    composite_score = to_float_list(comp, 3)

    # Regime labels (heuristic per point, using aligned tails)
    eff_tail = tail(eff)
    bw_tail  = tail(bw)
    adx_tail = tail(adx)

    def _label_point(idx):
        # normalize helpers
        e = eff_tail[idx] if idx < len(eff_tail) else 0.0
        b = bw_tail[idx]  if idx < len(bw_tail) else 0.0
        a = adx_tail[idx] if idx < len(adx_tail) else 0.0
        t = trend_score[idx]
        v = volatility_score[idx]
        # thresholds are heuristic and scale-agnostic via ranks
        if e >= np.median(eff_tail) and a >= np.median(adx_tail) and t >= 50:
            return "trending"
        if v >= np.percentile(volatility_score, 66) and e <= np.percentile(eff_tail, 33):
            return "volatile"
        if v <= np.percentile(volatility_score, 33) and t <= 40:
            return "quiet"
        return "range_bound"

    regime_labels = [_label_point(i) for i in range(k)]

    # Transition matrix from a longer recent history (if available)
    regimes_full = []
    eff_full = eff.dropna().values.tolist()
    adx_full = pd.Series(adx).dropna().values.tolist()
    bw_full  = pd.Series(bw).dropna().values.tolist()
    n = min(len(eff_full), len(adx_full), len(bw_full))
    if n >= 80:
        # coarse labels over longer span
        E = eff_full[-n:]
        A = adx_full[-n:]
        V = pd.Series(volatility_score_full).dropna().values[-n:] if len(volatility_score_full)>=n else [0]*n
        for i in range(n):
            if E[i] >= np.median(E) and A[i] >= np.median(A):
                regimes_full.append("trending")
            elif V[i] >= np.percentile(V, 66) and E[i] <= np.percentile(E, 33):
                regimes_full.append("volatile")
            elif V[i] <= np.percentile(V, 33):
                regimes_full.append("quiet")
            else:
                regimes_full.append("range_bound")

    states = ["trending","range_bound","volatile","quiet"]
    trans = {s:{t:0.0 for t in states} for s in states}
    if len(regimes_full) >= 30:
        for i in range(1, len(regimes_full)):
            prev, nxt = regimes_full[i-1], regimes_full[i]
            if prev in trans and nxt in trans[prev]:
                trans[prev][nxt] += 1.0
        # row normalize
        for s in states:
            row_sum = sum(trans[s].values())
            if row_sum > 0:
                for t in states:
                    trans[s][t] = round(trans[s][t] / row_sum, 3)

    # ---------- assemble payloads ----------
    lists = {
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volatility_score": volatility_score,
        "flow_score": flow_score,
        "composite_score": composite_score,
        "regime_labels": regime_labels
    }

    helpers = {
        "weights": weights,
        "composite_last": last(composite_score),
        "trend_last": last(trend_score),
        "momentum_last": last(momentum_score),
        "volatility_last": last(volatility_score),
        "flow_last": last(flow_score),
        "transition_matrix": trans
    }

    # final JSON chunks
    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps(helpers, separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k, "weights": weights},
                              separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    prompt = f"""
STRICT JSON ONLY. No markdown. No extra keys.
If insufficient: {{"status":"insufficient_data","reason":"<reason>"}}.

SCHEMA:
{{
  "status":"ok",
  "header": {header_json},
  "summary": {{
    "market_regime":"trending"|"range_bound"|"volatile"|"quiet"|"mixed",
    "composite_band":"very_weak"|"weak"|"neutral"|"strong"|"very_strong",
    "components":{{"trend":0..100,"momentum":0..100,"volatility":0..100,"flow":0..100}},
    "transition":{{"current":"<regime>","one_step":{{"trending":0..1,"range_bound":0..1,"volatile":0..1,"quiet":0..1}}}},
    "confidence":0..1,
    "explanation":"Plain-language (no indicator names).",
    "rationale":"Concise numeric reasoning (no indicator names)."
  }},
  "safety":{{"advice_compliance":"no_advice"}}
}}

RULES:
- Use ONLY numeric lists below.
- market_regime based on coherence among trend/momentum/volatility/flow + efficiency.
- composite_band from composite_score: <20 very_weak; 20–40 weak; 40–60 neutral; 60–80 strong; ≥80 very_strong.
- confidence <1.0, based on agreement only.

DATA:
{data_json}

FEATURES:
{helpers_json}

Return STRICT JSON only.
""".strip()

    return prompt



TOOLS = [trend_detection, momentum_strength, volatility_range, volume_flow,
         market_structure, liquidity_participation, risk_efficiency, regime_composite]


