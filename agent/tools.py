import json

from langchain.tools import tool
import numpy as np
import pandas as pd
import talib
from talib import MA_Type

from .utils import _fetch_finance_df_with_symbol, _last, _minmax01, \
      _tail, _to_float_list, _slope, _z


@tool("trend_detection")
def trend_detection(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Trend Detection* from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** provide advice.
    The downstream LLM must summarize **direction, strength, and momentum state** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Columns required:
        ["Open","High","Low","Close","Volume"] ordered by time ascending.

    Data semantics
    --------------
    - Lookback: 14 most recent sessions.
    - Array order: **oldest → newest**.
    - All values are decimals (≤ 3 dp).

    Computed features (last 14)
    ---------------------------
    - ema_5, ema_12, ema_26            : short/mid/long smoothed prices
    - ema12_minus_ema26                : mid–long spread
    - macd, macd_signal, macd_hist     : differential / reference / residual series
    - adx, plus_di, minus_di           : directional clarity proxies
    - cci                              : deviation-from-mean proxy
    Helper features: ema_spread_slope, macd_hist_slope, adx_slope, cci_z

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "direction":"up"|"down"|"sideways"|"mixed",
          "strength":"weak"|"moderate"|"strong",
          "momentum":"building"|"fading"|"stable",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Failure Modes
    -------------
    - If any required list has <14 points or non-numeric values, return:
      {"status":"insufficient_data","reason":"<reason>"}
    - Cache miss / bad schema → raise upstream error.

    Guardrails & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Recommend LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """
    k = 14
    df = _fetch_finance_df_with_symbol(symbol=symbol)
    close, high, low = df["Close"], df["High"], df["Low"]
    ema_5  = talib.EMA(close, 5)
    ema_12 = talib.EMA(close, 12)
    ema_26 = talib.EMA(close, 26)
    ema12_26 = ema_12 - ema_26

    macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
    adx      = talib.ADX(high, low, close, 14)
    plus_di  = talib.PLUS_DI(high, low, close, 14)
    minus_di = talib.MINUS_DI(high, low, close, 14)
    cci      = talib.CCI(high, low, close, 14)

    # ---------- assemble lists (last k points, rounded) ----------
    lists = {
        "ema_5": _tail(ema_5, k),
        "ema_12": _tail(ema_12, k),
        "ema_26": _tail(ema_26, k),
        "ema12_minus_ema26": _tail(ema12_26, k),
        "macd": _tail(macd, k),
        "macd_signal": _tail(macd_sig, k),
        "macd_hist": _tail(macd_hist, k),
        "adx": _tail(adx, k),
        "plus_di": _tail(plus_di, k),
        "minus_di": _tail(minus_di, k),
        "cci": _tail(cci, k),
    }

    # ---------- sanity checks ----------
    required = ["ema_5","ema_12","ema_26","ema12_minus_ema26",
                "macd","macd_signal","macd_hist","adx","plus_di","minus_di","cci"]
    for key in required:
        if len(lists[key]) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    # ensure native floats (no np.float64) and fixed precision
    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # ---------- helper features ----------
    features = {
        "ema_spread_slope": _slope(lists["ema12_minus_ema26"]),
        "macd_hist_slope":  _slope(lists["macd_hist"]),
        "adx_slope":        _slope(lists["adx"]),
        "cci_z":            _z(lists["cci"]),
    }
    
    # cast helpers to JSON-safe floats
    features_json = {
        "ema_spread_slope": float(features["ema_spread_slope"]),
        "macd_hist_slope":  float(features["macd_hist_slope"]),
        "adx_slope":        float(features["adx_slope"]),
        "cci_z":            _to_float_list(features["cci_z"], 3)
    }

    header = {"symbol": symbol, "lookback_points": k}

    # ---------- stable JSON blocks ----------
    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps(features_json, separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps(header, separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    prompt = (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Provide an objective description of price behavior. No advice, predictions, targets, or instructions.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include your reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any required list has fewer than {k} points, or values are non-numeric, respond:\n'
f'{{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "direction":"up"|"down"|"sideways"|"mixed",\n'
f'    "strength":"weak"|"moderate"|"strong",\n'
f'    "momentum":"building"|"fading"|"stable",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Short plain-language summary with no indicator names.",\n'
f'    "rationale":"One concise technical justification with no indicator names."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "ema_5": "5-period Exponential Moving Average",\n'
f'  "ema_12": "12-period Exponential Moving Average",\n'
f'  "ema_26": "26-period Exponential Moving Average",\n'
f'  "macd": "Momentum differential between short and long averages",\n'
f'  "macd_signal": "Smoothed momentum reference line",\n'
f'  "macd_hist": "Difference between momentum and reference line",\n'
f'  "adx": "Trend strength measure",\n'
f'  "plus_di": "Positive movement measure",\n'
f'  "minus_di": "Negative movement measure",\n'
f'  "cci": "Price deviation strength from mean"\n'
f'}}\n\n'
f'These mappings are only to clarify feature meaning — do not mention or restate them in your output.\n\n'
f'Each array lists the most recent 14 sequential values from oldest → newest.\n'
f'DECISION RULES\n'
f'- If most lists imply the same direction, set "direction" accordingly; otherwise "mixed".\n'
f'- If recent changes are small and contradictory, prefer "sideways" or "mixed".\n'
f'- Map strength by magnitude/consistency across lists (weak/moderate/strong).\n'
f'- Momentum reflects recent acceleration or deceleration (building/fading/stable).\n'
f'- Set "confidence" in [0.00,1.00] proportional to agreement; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.'
    )
    return prompt


@tool("momentum_strength")
def momentum_strength(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Momentum & Strength* from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** provide advice.
    The downstream LLM must summarize **short-term momentum, participation strength, and buy/sell pressure**
    using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns:
        ["Open","High","Low","Close","Volume"] in ascending time.

    Data semantics
    --------------
    - Lookback: last 14 **sessions**.
    - Array order: **oldest → newest**.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data.

    Computed features (last 14)
    ---------------------------
    - rsi                 : bounded momentum series
    - slowk, slowd        : short-horizon oscillator (library defaults, e.g., STOCH fastk=5, fastd=3, type=SMA)
    - roc                 : rate-of-change (speed)
    - mfi                 : volume-aware momentum
    Helper features: rsi_slope, mfi_slope, roc_slope, stoch_diff_slope (slowk−slowd), rsi_z, mfi_z.

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "momentum_state":"rising"|"falling"|"neutral"|"mixed",
          "strength":"weak"|"moderate"|"strong",
          "pressure":"buying"|"selling"|"balanced",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Failure Modes
    -------------
    - If any required list has <14 points, return:
      {"status":"insufficient_data","reason":"<reason>"}
    - Upstream errors: cache miss, corrupt payloads, or bad schema.

    Guardrails & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    - Handle zero/near-zero volume safely for MFI; ensure numeric coercion.
    """
    k = 14
    df = _fetch_finance_df_with_symbol(symbol=symbol)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi = _tail(talib.RSI(close, timeperiod=14), k)
    slowk, slowd = talib.STOCH(high, low, close)
    slowk, slowd = _tail(slowk, k), _tail(slowd, k)
    roc = _tail(talib.ROC(close), k)
    mfi = _tail(talib.MFI(high, low, close, vol), k)

    # ---------- lists ----------
    lists = {
        "rsi": rsi,
        "slowk": slowk,
        "slowd": slowd,
        "roc": roc,
        "mfi": mfi,
    }

    # ---------- sanity check ----------
    for key, val in lists.items():
        if len(val) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # ---------- helper features ----------
    features = {
        "rsi_slope": _slope(lists["rsi"]),
        "stoch_diff_slope": _slope(np.array(lists["slowk"]) - np.array(lists["slowd"])),
        "roc_slope": _slope(lists["roc"]),
        "mfi_slope": _slope(lists["mfi"]),
        "rsi_z": _z(lists["rsi"]),
        "mfi_z": _z(lists["mfi"]),
    }

    features_json = json.dumps({k:v if isinstance(v, list) else float(v) for k,v in features.items()},
                               separators=(",", ":"), ensure_ascii=False)
    data_json = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    header_json = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    prompt = (
f'You are a market analysis model.\n'
f"Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n"
f"Analyze recent price momentum and strength objectively. No advice, predictions, or targets.\n\n"
f"OUTPUT REQUIREMENTS\n"
f"- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n"
f"- Do not include your reasoning steps.\n"
f"- Numeric fields must be decimals with at most 3 digits after the point.\n\n"
f"IF DATA INSUFFICIENT\n"
f'- If any list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f"SCHEMA (must match exactly)\n"
f"{{\n"
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "momentum_state":"rising"|"falling"|"neutral"|"mixed",\n'
f'    "strength":"weak"|"moderate"|"strong",\n'
f'    "pressure":"buying"|"selling"|"balanced",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary of current momentum and strength (no indicator names).",\n'
f'    "rationale":"Concise reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f"}}\n\n"
f"Feature mapping:\n"
f'{{\n'
f'  "rsi": "Relative strength measure of market momentum",\n'
f'  "slowk": "Short-term momentum oscillator component",\n'
f'  "slowd": "Smoothed momentum reference",\n'
f'  "roc": "Rate of price change (speed of movement)",\n'
f'  "mfi": "Money flow strength combining price and volume"\n'
f'}}\n\n'
f"These mappings clarify feature meaning — do not restate them in your output.\n"
f"Each array lists the most recent {k} sequential values from oldest → newest.\n\n"
f"DECISION RULES\n"
f"- If momentum measures increase together, classify as 'rising'.\n"
f"- If they decline together, classify as 'falling'.\n"
f"- Strength depends on magnitude and agreement (weak/moderate/strong).\n"
f"- Pressure represents buyer vs seller dominance (buying/selling/balanced).\n"
f"- Confidence is proportional to overall consistency.\n\n"
f"NUMERIC DATA\n"
f"{data_json}\n\n"
f"HELPER FEATURES\n"
f"{features_json}\n\n"
f"Return only the JSON. Max 120 tokens."
    )
    return prompt


@tool("volatility_range")
def volatility_range(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Volatility & Range* from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** provide advice.
    The downstream LLM must summarize **volatility regime, absolute level, and current price location within a
    dynamic range** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns:
        ["Open","High","Low","Close","Volume"], time-ascending.

    Data semantics
    --------------
    - Lookback: last **14 sessions**.
    - Array order: **oldest → newest**.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data.
    - BBANDS configured with `MA_Type.T3` (documented for determinism).

    Computed features (last 14)
    ---------------------------
    - atr                     : average true-range (volatility magnitude)
    - bb_upper/middle/lower   : dynamic range envelope
    - stddev_5                : short-horizon dispersion
    Helper features:
    - band_width, band_width_slope
    - band_position (0=near lower band, 1=near upper band) derived from closes
    - atr_slope, stddev_slope
    - atr_z, stddev_z (normalized context)

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "volatility_regime":"expanding"|"contracting"|"stable"|"mixed",
          "volatility_level":"low"|"medium"|"high",
          "relative_position":"near_high"|"near_low"|"mid_range",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Failure Modes
    -------------
    - If any required list has <14 points, return:
      {"status":"insufficient_data","reason":"<reason>"}
    - Upstream errors: cache miss, corrupt payloads, or bad schema.
    - Guards: divide-by-zero protected for band width; NaNs rejected.

    Guardrails & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """
    k = 14
    df = _fetch_finance_df_with_symbol(symbol=symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    # ---------- indicators ----------
    atr = _tail(talib.ATR(high, low, close, timeperiod=14), k)
    upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
    upper, middle, lower = _tail(upper, k), _tail(middle, k), _tail(lower, k)
    stdv5 = _tail(talib.STDDEV(close, timeperiod=5, nbdev=1), k)

    # ---------- sanity checks ----------
    lists = {
        "atr": atr,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "stddev_5": stdv5,
    }
    for key, val in lists.items():
        if len(val) < k:
            return json.dumps({"status": "insufficient_data", "reason": f"{key}_len<{k}"})

    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # ---------- helper features ----------
    # Band width & position (relative placement of price within bands)
    bw = (np.array(lists["bb_upper"]) - np.array(lists["bb_lower"]))        # width
    # Use last k closes to compute relative position; align with lists length k
    recent_close = close.dropna().tail(k).values.astype(float)
    # Avoid divide-by-zero
    denom = np.maximum(bw, 1e-12)
    band_pos = np.clip((recent_close - np.array(lists["bb_lower"])) / denom, 0.0, 1.0)

    helpers = {
        "band_width": _to_float_list(bw, 3),
        "band_width_slope": _slope(bw),
        "band_position": _to_float_list(band_pos, 3),  # 0=near lower, 1=near upper
        "atr_slope": _slope(lists["atr"]),
        "stddev_slope": _slope(lists["stddev_5"]),
        "atr_z": _z(lists["atr"]),
        "stddev_z": _z(lists["stddev_5"]),
    }

    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps({k:(v if isinstance(v, list) else float(v)) for k,v in helpers.items()},
                              separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    prompt = (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Assess recent volatility and trading range objectively. No advice, predictions, targets, or instructions.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include your reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "volatility_regime":"expanding"|"contracting"|"stable"|"mixed",\n'
f'    "volatility_level":"low"|"medium"|"high",\n'
f'    "relative_position":"near_high"|"near_low"|"mid_range",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary of volatility and range (no indicator names).",\n'
f'    "rationale":"Concise reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "atr": "Average true range representing overall volatility magnitude",\n'
f'  "bb_upper/bb_middle/bb_lower": "Dynamic range envelope capturing price extremes and mean reversion zone",\n'
f'  "stddev_5": "Short-term statistical dispersion of closing prices"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- If dispersion and spread measures are increasing together, set "volatility_regime":"expanding"; if decreasing, "contracting"; else "stable" or "mixed".\n'
f'- Map "volatility_level" using recent magnitude of dispersion/spread vs its own history (low/medium/high).\n'
f'- "range_state" reflects where recent prices sit within the dynamic range (near_high/near_low/mid_range).\n'
f'- Confidence is proportional to agreement across measures; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.'
    )
    return prompt


@tool("volume_flow")
def volume_flow(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Volume & Flow* from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** provide advice.
    The downstream LLM must summarize **trading activity, participation strength, and net flow** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns:
        ["Open","High","Low","Close","Volume"], time-ascending.

    Data semantics
    --------------
    - Lookback: last **14 sessions**.
    - Array order: **oldest → newest**.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data handling.

    Computed features (last 14)
    ---------------------------
    - obv               : cumulative volume flow proxy
    - ad                : accumulation/distribution flow proxy
    - volume_ema_10/20  : short- and medium-horizon smoothed activity
    Helper features:
    - obv_slope, ad_slope, obv_z, ad_z
    - volume_ratio = volume_ema_10 / volume_ema_20 (guarded for divide-by-zero)

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "volume_trend":"increasing"|"decreasing"|"stable"|"mixed",
          "flow_bias":"inflow"|"outflow"|"balanced",
          "participation_strength":"weak"|"moderate"|"strong",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Failure Modes
    -------------
    - If any required list has <14 points, return:
      {"status":"insufficient_data","reason":"<reason>"}
    - Upstream errors: cache miss, corrupt payloads, or bad schema.
    - Guards: zero/near-zero volume and divide-by-zero safely handled.

    Guardrails & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """
    k: int = 14
    df = _fetch_finance_df_with_symbol(symbol=symbol)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    obv = _tail(talib.OBV(close, vol), k)
    ad  = _tail(talib.AD(high, low, close, vol), k)
    volume_ema_10 = _tail(talib.EMA(vol, timeperiod=10), k)
    volume_ema_20 = _tail(talib.EMA(vol, timeperiod=20), k)

    # ---------- assemble lists ----------
    lists = {
        "obv": obv,
        "ad": ad,
        "volume_ema_10": volume_ema_10,
        "volume_ema_20": volume_ema_20,
    }

    # ---------- sanity check ----------
    for key, val in lists.items():
        if len(val) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})
    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # ---------- helper features ----------
    features = {
        "obv_slope": _slope(lists["obv"]),
        "ad_slope": _slope(lists["ad"]),
        "volume_ratio": round(lists["volume_ema_10"][-1] / (lists["volume_ema_20"][-1] + 1e-12), 3),
        "obv_z": _z(lists["obv"]),
        "ad_z": _z(lists["ad"]),
    }

    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps({k:(v if isinstance(v, list) else float(v)) for k,v in features.items()},
                              separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    prompt = (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Assess trading activity, participation strength, and capital flow objectively. No advice, predictions, or targets.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "volume_trend":"increasing"|"decreasing"|"stable"|"mixed",\n'
f'    "flow_bias":"inflow"|"outflow"|"balanced",\n'
f'    "participation_strength":"weak"|"moderate"|"strong",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary of market volume and flow (no indicator names).",\n'
f'    "rationale":"Concise reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "obv": "Cumulative volume measure confirming price-direction flow",\n'
f'  "ad": "Volume-weighted accumulation/distribution line capturing inflow vs outflow",\n'
f'  "volume_ema_10/20": "Smoothed short- and medium-term trading activity averages"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- If cumulative volume measures rise together, classify as inflow with increasing participation.\n'
f'- If they decline together, classify as outflow with fading participation.\n'
f'- Compare short vs long volume averages to determine rising or falling activity.\n'
f'- Confidence is proportional to consistency among signals; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.'
    )
    return prompt


@tool("market_structure")
def market_structure(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Market Structure & Price Action* from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** analyze or advise.
    The downstream LLM must summarize **structure, directional bias, price location within range, and key levels** using
    numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns:
        ["Open","High","Low","Close","Volume"], time-ascending.

    Data semantics
    --------------
    - Lookback: last **14 sessions**.
    - Array order: **oldest → newest**.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data handling.

    What it computes (last 14)
    --------------------------
    - Local swings: recent swing highs/lows (neighborhood window=3)
    - Range & position: rolling high–low width; normalized price_position in [0,1] (0=near low, 1=near high)
    - Candle shape: body_ratio, upper_wick_ratio, lower_wick_ratio (0..1)
    - Gaps: gap_up/gap_down flags vs prior close
    - Efficiency: net move / path length (0..1) for directionality
    - Key levels: histogram-clustered support/resistance from recent swing points (top 3 each)
    Helper features: range_width_slope, price_position_last, efficiency_last, gap rates, swing counts, z-scores.

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "structure":"trend"|"range"|"transition"|"mixed",
          "bias":"up"|"down"|"neutral"|"mixed",
          "price_position":"near_high"|"near_low"|"mid_range",
          "key_levels":{"support":[number,...],"resistance":[number,...]},
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Decision rules (guidance to the LLM)
    ------------------------------------
    - Structure: higher efficiency with widening range and directional swing sequence ⇒ “trend”; low efficiency with
      tight, mean-reverting range ⇒ “range”; conflicting signals ⇒ “transition” or “mixed”.
    - Bias: favor the side with more recent higher highs vs lower lows, upward vs downward price_position drift,
      and body_ratio alignment; gaps can reinforce bias if consistent.
    - Price location: map last price_position to “near_high” (≥0.66), “mid_range” (0.33–0.66), or “near_low” (<0.33).
    - Confidence: proportional to agreement/magnitude across features; never 1.00.

    Failure Modes
    -------------
    - If any required list has <14 points, return:
      {"status":"insufficient_data","reason":"<reason>"}
    - Guards: divide-by-zero protected for range width; NaNs rejected; clustering uses densest histogram bins.

    Compliance & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """
    df = _fetch_finance_df_with_symbol(symbol=symbol)
    k = 14

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    # --- Local swings (window=3) -> 1 for swing, 0 otherwise
    swing_high_mask = (h.shift(1) < h) & (h.shift(-1) < h)
    swing_low_mask  = (l.shift(1) > l) & (l.shift(-1) > l)
    swing_high_vals = h.where(swing_high_mask)
    swing_low_vals  = l.where(swing_low_mask)

    # --- Range/position helpers
    recent_high = h.rolling(window=k, min_periods=1).max()
    recent_low  = l.rolling(window=k, min_periods=1).min()
    rng = recent_high - recent_low
    price_pos = np.clip((c - recent_low) / (rng.replace(0, np.nan)).fillna(1e-12), 0.0, 1.0)  # 0=low,1=high

    # --- Candle shape (body/wicks ratios)
    true_range = (h - l).replace(0, 1e-12)
    body = (c - o).abs()
    upper_wick = (h - np.maximum(o, c))
    lower_wick = (np.minimum(o, c) - l)
    body_ratio = (body / true_range).clip(0, 1)
    upper_wick_ratio = (upper_wick / true_range).clip(0, 1)
    lower_wick_ratio = (lower_wick / true_range).clip(0, 1)

    # --- Gaps vs prior close
    prev_close = c.shift(1)
    gap = o - prev_close
    gap_up = (gap > 0).astype(int)
    gap_down = (gap < 0).astype(int)

    # --- Efficiency ratio (price path directionality)
    net_change = (c - c.shift(k)).abs()
    path = (c.diff().abs().rolling(k).sum())
    efficiency = (net_change / path.replace(0, np.nan)).fillna(0.0).clip(0, 1)

    # --- Simple clustering of support/resistance from last ~100 swings
    def _cluster_levels(vals: pd.Series, n_levels: int = 3) -> list:
        x = vals.dropna().tail(100).values.astype(float)
        if x.size == 0:
            return []
        # histogram-based clustering: pick densest bins (20 bins)
        hist, edges = np.histogram(x, bins=20)
        idx = hist.argsort()[::-1][:n_levels]
        centers = [(edges[i] + edges[i+1]) / 2.0 for i in idx]
        centers = sorted([round(float(v), 3) for v in centers])
        return centers

    supports   = _cluster_levels(swing_low_vals, n_levels=3)
    resistances= _cluster_levels(swing_high_vals, n_levels=3)

    # --- Assemble lists (last k points)
    lists = {
        "highs":          _tail(h, k),
        "lows":           _tail(l, k),
        "closes":         _tail(c, k),
        "range_width":    _tail(rng, k),
        "price_position": _tail(price_pos, k),       # 0..1 within recent range
        "body_ratio":     _tail(body_ratio, k),      # 0..1
        "upper_wick_ratio": _tail(upper_wick_ratio, k),
        "lower_wick_ratio": _tail(lower_wick_ratio, k),
        "gap_up":         _tail(gap_up, k),          # 0/1
        "gap_down":       _tail(gap_down, k),        # 0/1
        "swing_highs":    _tail(swing_high_vals.fillna(np.nan), k), 
        "swing_lows":     _tail(swing_low_vals.fillna(np.nan), k),
        "efficiency":     _tail(efficiency, k),      # 0..1 (closer to 1 = directional)
    }

    # sanity check
    for key, arr in lists.items():
        if len(arr) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    # --- Helper features for the LLM
    helpers = {
        "range_width_slope": _slope(lists["range_width"]),
        "price_position_last": round(float(lists["price_position"][-1]), 3),
        "efficiency_last":     round(float(lists["efficiency"][-1]), 3),
        "gap_rate_up":   round(float(np.mean(lists["gap_up"])), 3),
        "gap_rate_down": round(float(np.mean(lists["gap_down"])), 3),
        "swing_high_count": int(np.isfinite(np.array(lists["swing_highs"], float)).sum()),
        "swing_low_count":  int(np.isfinite(np.array(lists["swing_lows"], float)).sum()),
        "support_levels": supports,
        "resistance_levels": resistances,
        "range_width_z": _z(lists["range_width"]),
        "body_ratio_z":  _z(lists["body_ratio"]),
    }

    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps(helpers, separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    return (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Assess market structure and price action objectively (structure, bias, key levels, and location within range). No advice, predictions, or targets.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "structure":"trend"|"range"|"transition"|"mixed",\n'
f'    "bias":"up"|"down"|"neutral"|"mixed",\n'
f'    "price_position":"near_high"|"near_low"|"mid_range",\n'
f'    "key_levels":{{"support":[number,...],"resistance":[number,...]}},\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary of recent market structure (no indicator names).",\n'
f'    "rationale":"Concise numeric reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}'
f'Feature mapping:\n'
f'{{\n'
f'  "highs/lows/closes": "Recent session extremes and settlements",\n'
f'  "range_width": "Rolling high–low distance capturing breadth of movement",\n'
f'  "price_position": "Normalized location within recent range (0=near low, 1=near high)",\n'
f'  "body_ratio": "Candle body size as a share of full session range (0..1)",\n'
f'  "upper_wick_ratio/lower_wick_ratio": "Upper/lower shadow size as share of range (0..1)",\n'
f'  "gap_up/gap_down": "Binary open vs prior close jumps (1=yes, 0=no)",\n'
f'  "swing_highs/swing_lows": "Local extrema values from neighborhood comparison",\n'
f'  "efficiency": "Directional efficiency (net move / path length, 0..1)"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- If cumulative volume measures rise together, classify as inflow with increasing participation.\n'
f'- If they decline together, classify as outflow with fading participation.\n'
f'- Compare short vs long volume averages to determine rising or falling activity.\n'
f'- Confidence is proportional to consistency among signals; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.')


@tool("liquidity_participation")
def liquidity_participation(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Liquidity & Participation Quality*
    from cached OHLCV and microstructure data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** analyze or advise.
    The downstream LLM must summarize **liquidity level, spread tightness, and participation quality** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns (time-ascending):
        ["Open","High","Low","Close","Volume"].
        Optional columns if available:
        - "Float" (shares float) for turnover ratio
        - "Trades" (count of executed prints) for average trade size
        - "Bid","Ask" for direct spread; otherwise a high–low proxy is used
        - "TickVolume" if you store it as a proxy for prints

    Data semantics
    --------------
    - Lookback: last **14 sessions**.
    - Array order: **oldest → newest**.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data handling.

    What it computes (last 14)
    --------------------------
    - turnover_ratio      : Volume / Float  (requires Float > 0)
    - dollar_volume       : Close * Volume  (USD notional proxy)
    - spread_proxy        : (Ask−Bid)/mid if Bid/Ask available; else (High−Low)/max(Close,1e-12)
    - volume_volatility   : rolling std of Volume (normalized by its mean)
    - avg_trade_size      : Volume / Trades (if Trades>0); else uses Volume / TickVolume if available
    Helper features: slopes (turnover_slope, spread_slope, dv_slope), last values, and z-scores.

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "liquidity_level":"low"|"medium"|"high",
          "spread_state":"tight"|"normal"|"wide",
          "participation_quality":"institutional"|"retail"|"mixed"|"insufficient",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Decision rules (guidance to the LLM)
    ------------------------------------
    - Liquidity level: use dollar_volume and turnover_ratio direction/magnitude (higher ⇒ “high”).
    - Spread state: smaller, consistent spread_proxy ⇒ “tight”; larger and widening ⇒ “wide”.
    - Participation quality: larger & rising avg_trade_size with steady turnover ⇒ “institutional”;
      very small/volatile avg_trade_size ⇒ “retail”; mixed evidence ⇒ “mixed”; missing prints ⇒ “insufficient”.
    - Confidence: proportional to agreement/magnitude across features; never 1.00.

    Failure Modes
    -------------
    - If **dollar_volume**, **spread_proxy**, or **volume_volatility** have <14 points →:
      {"status":"insufficient_data","reason":"core_features_len<14"}
    - turnover_ratio and avg_trade_size may be absent; we still proceed, but mark in helpers and allow
      the LLM to return “insufficient” for participation_quality if needed.
    - Guards: divide-by-zero for Float/Trades, NaNs rejected, bid/ask fallback to high–low proxy.

    Compliance & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """

    k = 14
    df = _fetch_finance_df_with_symbol(symbol=symbol)  # your cache fetcher

    # Required fields
    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    vol   = pd.to_numeric(df["Volume"], errors="coerce")

    # Optional fields
    flt   = pd.to_numeric(df.get("Float", pd.Series(index=df.index, dtype=float)), errors="coerce")
    trades = pd.to_numeric(df.get("Trades", pd.Series(index=df.index, dtype=float)), errors="coerce")
    tickv  = pd.to_numeric(df.get("TickVolume", pd.Series(index=df.index, dtype=float)), errors="coerce")
    bid    = pd.to_numeric(df.get("Bid", pd.Series(index=df.index, dtype=float)), errors="coerce")
    ask    = pd.to_numeric(df.get("Ask", pd.Series(index=df.index, dtype=float)), errors="coerce")

    # --- Core features ---
    dollar_volume = close * vol

    # Spread proxy preference: direct bid/ask if available, else high-low/close
    has_ba = bid.notna().any() and ask.notna().any()
    if has_ba:
        mid = ((bid + ask) / 2.0).replace(0, np.nan)
        spread_proxy = ((ask - bid) / mid).replace([np.inf, -np.inf], np.nan)
    else:
        denom = np.maximum(close.replace(0, np.nan), 1e-12)
        spread_proxy = ((high - low) / denom).replace([np.inf, -np.inf], np.nan)

    # Volume volatility (normalized sigma)
    vol_sigma = vol.rolling(20, min_periods=5).std()
    vol_mu    = vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    volume_volatility = (vol_sigma / vol_mu).replace([np.inf, -np.inf], np.nan)

    # Turnover ratio (optional)
    turnover_ratio = None
    if flt.notna().sum() > 0:
        denom = flt.replace(0, np.nan)
        turnover_ratio = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # Avg trade size (optional)
    avg_trade_size = None
    if trades.notna().sum() > 0:
        denom = trades.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)
    elif tickv.notna().sum() > 0:
        denom = tickv.replace(0, np.nan)
        avg_trade_size = (vol / denom).replace([np.inf, -np.inf], np.nan)

    # Assemble mandatory lists
    lists = {
        "dollar_volume": _tail(dollar_volume),
        "spread_proxy":  _tail(spread_proxy),
        "volume_volatility": _tail(volume_volatility),
    }

    # Optional lists
    if turnover_ratio is not None:
        lists["turnover_ratio"] = _tail(turnover_ratio)
    if avg_trade_size is not None:
        lists["avg_trade_size"] = _tail(avg_trade_size)

    # Sanity: core features must be present
    core_keys = ["dollar_volume", "spread_proxy", "volume_volatility"]
    for key in core_keys:
        if len(lists.get(key, [])) < k:
            return json.dumps({"status":"insufficient_data","reason":"core_features_len<14"})

    # Round/float-cast
    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # Helpers
    helpers = {
        "dv_slope": _slope(lists["dollar_volume"]),
        "spread_slope": _slope(lists["spread_proxy"]),
        "volvol_slope": _slope(lists["volume_volatility"]),
        "dv_z": _z(lists["dollar_volume"]),
        "spread_z": _z(lists["spread_proxy"]),
        "volvol_z": _z(lists["volume_volatility"]),
        "turnover_slope": _slope(lists["turnover_ratio"]) if "turnover_ratio" in lists else None,
        "avg_trade_size_slope": _slope(lists["avg_trade_size"]) if "avg_trade_size" in lists else None,
        "has_turnover": int("turnover_ratio" in lists),
        "has_avg_trade_size": int("avg_trade_size" in lists),
    }

    # JSON blocks
    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps({k:(v if not isinstance(v, np.floating) else float(v)) for k,v in helpers.items()},
                              separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    # Prompt
    return (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Assess liquidity level, spread tightness, and participation quality objectively. No advice, predictions, or targets.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any core list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "liquidity_level":"low"|"medium"|"high",\n'
f'    "spread_state":"tight"|"normal"|"wide",\n'
f'    "participation_quality":"institutional"|"retail"|"mixed"|"insufficient",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary (no indicator names).",\n'
f'    "rationale":"Concise numeric reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "dollar_volume":"Close×Volume (notional trading activity proxy)",\n'
f'  "turnover_ratio":"Volume divided by free float (trading intensity)",\n'
f'  "spread_proxy":"Quoted spread over mid or high–low/Close proxy (tightness)",\n'
f'  "volume_volatility":"Normalized standard deviation of Volume (crowding/quiet)",\n'
f'  "avg_trade_size":"Volume per trade (institutional vs retail footprint proxy)"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- Liquidity level: use dollar_volume trend/level and turnover_ratio (if present).\n'
f'- Spread state: map spread_proxy magnitude and slope to tight/normal/wide.\n'
f'- Participation quality: infer from avg_trade_size level/consistency; if missing, return "insufficient".\n'
f'- Confidence is proportional to agreement and magnitude across features; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.'
    )


@tool("risk_efficiency")
def risk_efficiency(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Risk & Efficiency Metrics*
    from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. This tool returns a **prompt string**; it does **not** analyze or advise.
    The downstream LLM must summarize **risk regime, downside pressure, and price path efficiency** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns (time-ascending):
        ["Open","High","Low","Close","Volume"].

    Data semantics
    --------------
    - Lookback: last **14 sessions** for output lists (older → newer).
    - Computations use a rolling window W=20 (or larger where noted), then we emit the **last 14** values.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN triggers insufficient-data handling.

    What it computes (rolling, emit last 14)
    ----------------------------------------
    - sharpe_rolling     : reward-to-variability proxy (mean/σ of returns, scaled √252)
    - sortino_rolling    : reward-to-downside-variability proxy (downside σ, scaled √252)
    - mdd_rolling        : maximum drawdown magnitude within window (positive number)
    - ulcer_index        : sqrt(mean(drawdown^2)) within window
    - eff_ratio          : |ΔPrice over window| / sum(|ΔPrice daily|) (0..1)
    - var_95             : 95% historical Value-at-Risk (positive = loss magnitude)
    - es_95              : 95% historical Expected Shortfall (positive = loss magnitude)
    Helper features: slopes of sharpe/eff_ratio/mdd/ulcer, last values, and z-scores.

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14},
        "summary":{
          "risk_regime":"muted"|"normal"|"elevated"|"mixed",
          "efficiency_state":"efficient"|"noisy"|"mixed",
          "downside_risk":"low"|"medium"|"high",
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Decision rules (guidance to the LLM)
    ------------------------------------
    - Risk regime: map dispersion/drawdown mix (var_95/es_95/ulcer/mdd) — lower ⇒ “muted”, higher ⇒ “elevated”.
    - Efficiency: higher eff_ratio with supportive reward/σ ⇒ “efficient”; low eff_ratio with choppy path ⇒ “noisy”.
    - Downside risk: use the recent magnitude of mdd, es_95, and ulcer (last values) vs their own history.
    - Confidence: proportional to agreement/magnitude across features; never 1.00.

    Failure Modes
    -------------
    - If any core list has <14 points, return: {"status":"insufficient_data","reason":"core_features_len<14"}.
    - Guards: divide-by-zero protections, NaNs rejected, windows require sufficient samples.

    Compliance & Runtime Hints
    --------------------------
    - No trading signals, predictions, targets, or instructions.
    - Do not mention indicator names in explanation/rationale (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 120.
    """
    import json
    import numpy as np
    import pandas as pd

    k = 14
    W = 20
    ANNUAL = np.sqrt(252.0)

    df = _fetch_finance_df_with_symbol(symbol=symbol)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)

    # --- sanity on length ---
    if close.dropna().shape[0] < (W + k):
        return json.dumps({"status":"insufficient_data","reason":"not_enough_prices"})

    # --- returns ---
    ret = close.pct_change()

    # rolling mean/std
    mu = ret.rolling(W, min_periods=W).mean()
    sd = ret.rolling(W, min_periods=W).std()

    # downside std for Sortino
    neg = ret.clip(upper=0.0)
    dsd = neg.rolling(W, min_periods=W).std()

    # Sharpe/Sortino (risk-free assumed 0 for prompt context)
    sharpe = (mu / sd.replace(0, np.nan)) * ANNUAL
    sortino = (mu / dsd.replace(0, np.nan)) * ANNUAL

    # Wealth index & drawdowns (rolling MDD and Ulcer)
    wealth = (1.0 + ret.fillna(0.0)).cumprod()
    roll_max = wealth.rolling(W, min_periods=W).max()
    dd = wealth / roll_max.replace(0, np.nan) - 1.0  # ≤ 0
    mdd = dd.rolling(W, min_periods=W).min().abs()   # positive magnitude
    ulcer = np.sqrt((dd.pow(2)).rolling(W, min_periods=W).mean())

    # Efficiency ratio (ER): |Δ over W| / sum(|daily Δ|)
    delta = close.diff()
    path_len = delta.abs().rolling(W, min_periods=W).sum()
    net_move = (close - close.shift(W)).abs()
    er = (net_move / path_len.replace(0, np.nan)).clip(0, 1)

    # Historical VaR / ES at 95% over W
    def _var_es_95(x: pd.Series):
        x = x.dropna()
        if x.shape[0] < W:
            return np.nan, np.nan
        q = np.quantile(x, 0.05)
        es = x[x <= q].mean() if (x <= q).any() else q
        # Return positive loss magnitudes
        return float(-q), float(-es)

    var_95 = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(pd.Series(s))[0], raw=False)
    es_95  = ret.rolling(W, min_periods=W).apply(lambda s: _var_es_95(pd.Series(s))[1], raw=False)

    # --- assemble lists (core) ---
    lists = {
        "sharpe_rolling": _tail(sharpe),
        "sortino_rolling": _tail(sortino),
        "mdd_rolling": _tail(mdd),
        "ulcer_index": _tail(ulcer),
        "eff_ratio": _tail(er),
        "var_95": _tail(var_95),
        "es_95": _tail(es_95),
    }

    # core completeness check
    for key, arr in lists.items():
        if len(arr) < k:
            return json.dumps({"status":"insufficient_data","reason":f"{key}_len<{k}"})

    # ensure native floats and precision
    lists = {k_: _to_float_list(v, 3) for k_, v in lists.items()}

    # helper features for LLM
    helpers = {
        "sharpe_slope": _slope(lists["sharpe_rolling"]),
        "eff_ratio_slope": _slope(lists["eff_ratio"]),
        "mdd_slope": _slope(lists["mdd_rolling"]),
        "ulcer_slope": _slope(lists["ulcer_index"]),
        "var95_last": float(lists["var_95"][-1]),
        "es95_last": float(lists["es_95"][-1]),
        "eff_ratio_last": float(lists["eff_ratio"][-1]),
        "sharpe_z": _z(lists["sharpe_rolling"]),
        "ulcer_z": _z(lists["ulcer_index"]),
    }

    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps(helpers, separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k}, separators=(",", ":"), ensure_ascii=False)

    # --- final prompt ---
    return (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Assess risk regime, downside risk, and price-path efficiency objectively. No advice, predictions, targets, or instructions.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "risk_regime":"muted"|"normal"|"elevated"|"mixed",\n'
f'    "efficiency_state":"efficient"|"noisy"|"mixed",\n'
f'    "downside_risk":"low"|"medium"|"high",\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary (no indicator names).",\n'
f'    "rationale":"Concise numeric reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "sharpe_rolling":"Rolling reward-to-variability (scaled)",\n'
f'  "sortino_rolling":"Rolling reward-to-downside-variability (scaled)",\n'
f'  "mdd_rolling":"Rolling maximum drawdown magnitude within window",\n'
f'  "ulcer_index":"Rolling square-root mean of drawdown squared",\n'
f'  "eff_ratio":"Directional efficiency (net move / path length, 0..1)",\n'
f'  "var_95":"95% historical Value-at-Risk (loss magnitude)",\n'
f'  "es_95":"95% Expected Shortfall (average loss beyond VaR)"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- Risk regime: use var_95, es_95, ulcer_index, and mdd_rolling magnitudes and directions.\n'
f'- Efficiency: map eff_ratio level/slope alongside reward/variability context.\n'
f'- Downside risk: emphasize es_95 and mdd_rolling recency; ulcer_index corroborates persistence.\n'
f'- Confidence is proportional to agreement and magnitude across features; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 120 tokens.'
    )


@tool("regime_composite")
def regime_composite(symbol: str) -> str:
    """
    Build a **strict-JSON analysis prompt** for evaluating a stock’s *Regime & Composite Scoring*
    from cached OHLCV data.

    Purpose
    -------
    For autonomous/semi-autonomous AI agents. Returns a **prompt string**; it does **not** analyze or advise.
    The downstream LLM must summarize **market regime** and an **interpretable composite score** using numeric series only.

    Inputs
    ------
    symbol : str
        Ticker used to fetch a cached pandas.DataFrame (Redis/RAG). Required columns (time-ascending):
        ["Open","High","Low","Close","Volume"].

    Data semantics
    --------------
    - Lookback for output lists: **14 sessions** (oldest → newest).
    - Internal roll windows: typically 14–20 for feature construction.
    - Values are decimals (≤ 3 dp). Non-numeric/NaN → insufficient-data handling.

    What it computes (emit last 14)
    --------------------------------
    Component sub-scores in **0–100**:
      - trend_score     : from EMA(12–26) spread slope + ADX level
      - momentum_score  : from RSI z, ROC slope, and STOCH (slowK−slowD) slope
      - volatility_score: from ATR z and Bollinger band-width slope
      - flow_score      : from OBV/AD slopes and short/long volume ratio
    Composite:
      - composite_score : weighted blend (default weights: trend=0.3, momentum=0.3, volatility=0.2, flow=0.2)
    Regime labeling (per point):
      - regime_labels   : "trending" | "range_bound" | "volatile" | "quiet"
        (heuristics combine efficiency, volatility, and trend breadth)

    Helper features
    ---------------
    - current weights, slopes, z-scores, and last values for each component
    - simple one-step **transition probabilities** estimated from a rolling regime series (4×4 row-normalized)

    Expected Output (LLM response)
    ------------------------------
    STRICT JSON matching exactly:
      {
        "status":"ok",
        "header":{"symbol":<str>,"lookback_points":14,"weights":{"trend":0.3,"momentum":0.3,"volatility":0.2,"flow":0.2}},
        "summary":{
          "market_regime":"trending"|"range_bound"|"volatile"|"quiet"|"mixed",
          "composite_band":"very_weak"|"weak"|"neutral"|"strong"|"very_strong",
          "components":{"trend":0..100,"momentum":0..100,"volatility":0..100,"flow":0..100},
          "transition":{"current": "<regime>", "one_step":{"trending":0..1,"range_bound":0..1,"volatile":0..1,"quiet":0..1}},
          "confidence":0..1,
          "explanation":"Plain-language summary (no indicator names).",
          "rationale":"Concise numeric reasoning (no indicator names)."
        },
        "safety":{"advice_compliance":"no_advice"}
      }
    If the first output violates the schema, the LLM must immediately return a corrected STRICT JSON.

    Decision rules (guidance to the LLM)
    ------------------------------------
    - Market regime:
        * high trend_score + above-average efficiency ⇒ “trending”
        * low trend_score + tight volatility_score ⇒ “range_bound”
        * high volatility_score with low efficiency ⇒ “volatile”
        * low volatility_score and muted movement ⇒ “quiet”
      Mixed/conflicting evidence ⇒ “mixed”.
    - Composite band: map composite_score → very_weak(<20), weak(20–40), neutral(40–60), strong(60–80), very_strong(≥80).
    - Confidence ∝ agreement and magnitude across components; never 1.00.

    Failure Modes
    -------------
    - If any required list has <14 points → {"status":"insufficient_data","reason":"<reason>"}.
    - Guards: divide-by-zero protection; NaNs rejected; transition matrix requires enough regime history.

    Compliance & Runtime Hints
    --------------------------
    - No signals, predictions, or instructions.
    - Don’t mention indicator names in explanations (feature mapping is for context only).
    - Suggested LLM params: temperature ≤ 0.3, max_tokens ≈ 150.
    """

    k = 14
    W = 20  # internal rolling window
    weights = {"trend": 0.3, "momentum": 0.3, "volatility": 0.2, "flow": 0.2}

    df = _fetch_finance_df_with_symbol(symbol=symbol)
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
        _minmax01(macd_hist) + 
        _minmax01(adx) + 
        _minmax01(spread)
    )
    trend_score_full = np.round(100 * np.array(_minmax01(pd.Series(trend_raw).rolling(W).mean())), 3).tolist()

    # Momentum: RSI z (abs/level), ROC slope, STOCH diff slope
    stoch_diff = (slowk - slowd)
    mom_raw = (
        _minmax01(rsi) + 
        _minmax01(roc) + 
        _minmax01(stoch_diff)
    )
    momentum_score_full = np.round(100 * np.array(_minmax01(pd.Series(mom_raw).rolling(W).mean())), 3).tolist()

    # Volatility: ATR z + BW slope (higher ⇒ more volatile)
    vol_raw = (
        _minmax01(atr) + 
        _minmax01(bw)
    )
    volatility_score_full = np.round(100 * np.array(_minmax01(pd.Series(vol_raw).rolling(W).mean())), 3).tolist()

    # Flow: OBV/AD slope + volume ratio (10/20)
    vol_ratio = (vol_ema10 / (vol_ema20.replace(0, np.nan))).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    flow_raw = (
        _minmax01(obv) + 
        _minmax01(ad) + 
        _minmax01(vol_ratio)
    )
    flow_score_full = np.round(100 * np.array(_minmax01(pd.Series(flow_raw).rolling(W).mean())), 3).tolist()

    # Synchronized tail (last 14)
    trend_score     = _tail(pd.Series(trend_score_full))
    momentum_score  = _tail(pd.Series(momentum_score_full))
    volatility_score= _tail(pd.Series(volatility_score_full))
    flow_score      = _tail(pd.Series(flow_score_full))

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
    composite_score = _to_float_list(comp, 3)

    # Regime labels (heuristic per point, using aligned tails)
    eff_tail = _tail(eff)
    bw_tail  = _tail(bw)
    adx_tail = _tail(adx)

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
        "composite_last": _last(composite_score),
        "trend_last": _last(trend_score),
        "momentum_last": _last(momentum_score),
        "volatility_last": _last(volatility_score),
        "flow_last": _last(flow_score),
        "transition_matrix": trans
    }

    # final JSON chunks
    data_json    = json.dumps(lists, separators=(",", ":"), ensure_ascii=False)
    helpers_json = json.dumps(helpers, separators=(",", ":"), ensure_ascii=False)
    header_json  = json.dumps({"symbol": symbol, "lookback_points": k, "weights": weights},
                              separators=(",", ":"), ensure_ascii=False)

    # ---------- final prompt ----------
    return (
f'You are a market analysis model.\n'
f'Use ONLY the numeric lists provided. Do NOT mention technical indicator names.\n'
f'Summarize market regime and an interpretable composite score objectively. No advice, predictions, or targets.\n\n'
f'OUTPUT REQUIREMENTS\n'
f'- Respond with STRICT JSON using double quotes only, no markdown, no extra keys.\n'
f'- Do not include reasoning steps.\n'
f'- Numeric fields must be decimals with at most 3 digits after the point.\n\n'
f'IF DATA INSUFFICIENT\n'
f'- If any required list has fewer than {k} points, respond: {{"status":"insufficient_data","reason":"<reason>"}}\n\n'
f'SCHEMA (must match exactly)\n'
f'{{\n'
f'  "status":"ok",\n'
f'  "header": {header_json},\n'
f'  "summary": {{\n'
f'    "market_regime":"trending"|"range_bound"|"volatile"|"quiet"|"mixed",\n'
f'    "composite_band":"very_weak"|"weak"|"neutral"|"strong"|"very_strong",\n'
f'    "components":{{"trend":0..100,"momentum":0..100,"volatility":0..100,"flow":0..100}},\n'
f'    "transition":{{"current":"<regime>","one_step":{{"trending":0..1,"range_bound":0..1,"volatile":0..1,"quiet":0..1}}}},\n'
f'    "confidence":0..1,\n'
f'    "explanation":"Plain-language summary (no indicator names).",\n'
f'    "rationale":"Concise numeric reasoning (no indicator names)."\n'
f'  }},\n'
f'  "safety":{{"advice_compliance":"no_advice"}}\n'
f'}}\n\n'
f'Feature mapping:\n'
f'{{\n'
f'  "trend_score":"Composite of spread/strength slopes (0–100)",\n'
f'  "momentum_score":"Composite of short-horizon acceleration metrics (0–100)",\n'
f'  "volatility_score":"Composite of dispersion/spread dynamics (0–100)",\n'
f'  "flow_score":"Composite of participation/flow dynamics (0–100)",\n'
f'  "composite_score":"Weighted blend of components (0–100)",\n'
f'  "regime_labels":"Per-point regime classification over the lookback"\n'
f'}}\n\n'
f'These mappings clarify feature meaning — do not restate them in your output.\n'
f'Each array lists the most recent {k} sequential values from oldest → newest.\n\n'
f'DECISION RULES\n'
f'- Market regime from component coherence and efficiency/dispersion mix; inconsistent evidence ⇒ "mixed".\n'
f'- Composite band from composite_score cutoffs: <20 very_weak; 20–40 weak; 40–60 neutral; 60–80 strong; ≥80 very_strong.\n'
f'- Confidence is proportional to cross-component agreement; never 1.00.\n\n'
f'NUMERIC DATA\n'
f'{data_json}\n\n'
f'HELPER FEATURES\n'
f'{helpers_json}\n\n'
f'Return only the JSON. Max 150 tokens.'
    )