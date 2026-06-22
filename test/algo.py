#!/usr/bin/env python3
"""
================================================================================
 SMC + OB + ChanEx — 1 Lot Strict Chandelier Strategy + Realistic Backtester
 (FIXED: safety, credentials, and backtest premium realism — see CHANGELOG)
================================================================================

CHANGELOG vs. original
-----------------------
1. SAFETY: `--live` flag previously did nothing — running the script with NO
   flags at all (`python algo_bot.py`) silently went live with real orders.
   Fixed: live trading now requires `--live` AND a typed confirmation phrase.
   No flags = paper mode, by default.
2. SECURITY: hardcoded DhanHQ client ID + access token removed from source.
   Now read from DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN environment variables.
   If a real token was ever pasted into this file, treat it as compromised
   and regenerate it from the Dhan developer console.
3. BACKTEST REALISM: the backtester used to enter every single trade at a
   hardcoded premium of ₹150 with a flat 0.5 delta, regardless of strike,
   moneyness, IV, or time decay — meaning every "win rate" it ever printed
   was meaningless. It's replaced with a Black-Scholes model (flat IV
   assumption, real time decay, strike selected the same way the live bot
   selects strikes). This is still an approximation (no real historical
   IV/bid-ask), not a guarantee of live performance — see
   DhanBroker.get_expired_option_candles() for a path to real historical
   option data if you want to go further than the model.
4. New optional ATR-based trailing stop (USE_ATR_TRAILING_SL), off by
   default, as a volatility-adjusted alternative to the fixed 20-point stop.
5. Backtest report now also prints profit factor, average win/loss, and
   expectancy per trade — win rate alone doesn't tell you if a strategy is
   profitable.

NOTE ON "70-80% WIN RATE": this code does not force any particular win rate,
and won't be made to. A backtest's job is to tell you the truth about a
strategy, not to be tuned until it reports a number you want — that's just
fabricating results, and trading on fabricated backtest numbers loses real
money. What you CAN legitimately do: validate the strategy logic on a large
enough sample (run --backtest 60 or more), look at win rate *together with*
average win/loss and profit factor (not in isolation), and iterate on entry
filters / exit structure from there.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG  -- edit these or override via environment variables / CLI flags
# ==============================================================================

class Config:
    # ---- Safety switch -------------------------------------------------
    DRY_RUN: bool = True

    # ---- DhanHQ credentials --------------------------------------------
    # SECURITY: do NOT hardcode real credentials in source. Set these as
    # environment variables before running, e.g.:
    #   export DHAN_CLIENT_ID="1110569990"
    #   export DHAN_ACCESS_TOKEN="your_jwt_here"
    DHAN_CLIENT_ID: str = "1110569990"
    DHAN_ACCESS_TOKEN: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgyMTg2MzIwLCJpYXQiOjE3ODIwOTk5MjAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwNTY5OTkwIn0.LHn9fx3rowd5wq6TRxvA4wT4y-jqD_4_NtvIFbPR1rmARvFHP0EsTkRsATEihPAGm_xPcXjh09xLL73udEiH7Q"


    # ---- Target Configuration (STRICTLY NIFTY 50) ----------------------
    UNDERLYING: str = "NIFTY"
    UNDERLYING_SECURITY_ID: Dict[str, str] = {"NIFTY": "13", "BANKNIFTY": "25"}

    UNDERLYING_EXCHANGE_SEGMENT: str = "IDX_I"
    OPTION_EXCHANGE_SEGMENT: str = "NSE_FNO"

    # ---- Candle timeframe --------------------------------------------
    TIMEFRAME_MINUTES: int = 3
    POLL_SECONDS: int = 5

    # ---- Strategy parameters (Chandelier Exit Settings) ----------
    SWING_LENGTH: int = 25
    INTERNAL_LENGTH: int = 3
    OB_LOOKBACK: int = 3
    OB_FILTER_METHOD: str = "atr"
    OB_MITIGATION: str = "highlow"
    MAX_ACTIVE_OBS: int = 5

    CE_ATR_PERIOD: int = 1
    CE_ATR_MULT: float = 1.7
    CE_USE_CLOSE: bool = True

    # ---- RISK MANAGEMENT: Strict 1 Lot + Trailing SL -------------------
    CAPITAL: float = 100000.0
    MAX_DAILY_LOSS_PCT: float = 10.0
    MAX_TRADES_PER_DAY: int = 4

    TRAILING_SL_POINTS: float = 10.0
    TRAIL_WITH_CHANDELIER: bool = True
    SQUARE_OFF_TIME: dt.time = dt.time(15, 15)

    LOT_SIZE: Dict[str, int] = {"NIFTY": 65, "BANKNIFTY": 15}
    STRIKE_STEP: Dict[str, int] = {"NIFTY": 50, "BANKNIFTY": 100}

    # ---- Optional volatility-adjusted trailing stop (NEW, off by default)
    # Fixed-point trailing stops don't adapt to volatility regime: too tight
    # when the market is calm, too loose when it's wild. When enabled, the
    # trailing buffer becomes |option_delta| * underlying_ATR * ATR_TRAIL_MULT
    # instead of a flat TRAILING_SL_POINTS.
    USE_ATR_TRAILING_SL: bool = False
    ATR_TRAIL_MULT: float = 2.5

    # ---- Backtest option-premium model ---------------------------------
    # No historical option tick feed is wired in, so the backtester *models*
    # premiums with Black-Scholes + a flat IV assumption rather than a
    # hardcoded constant. Still an approximation — see
    # DhanBroker.get_expired_option_candles() for real historical data.
    BACKTEST_IV_ASSUMPTION: float = 0.13
    BACKTEST_RISK_FREE_RATE: float = 0.07
    BACKTEST_ASSUMED_DAYS_TO_EXPIRY: float = 3.0
    BACKTEST_STRIKE_SCAN_STEPS: int = 60

    # ---- Logging / persistence ------------------------------------------
    LOG_FILE: str = "algo_bot.log"
    TRADE_LOG_CSV: str = "trades_log.csv"
    STATE_FILE: str = "bot_state.json"

    # ---- Web dashboard ----------------------------------------------------
    DASHBOARD_ENABLED: bool = True
    DASHBOARD_HOST: str = "127.0.0.1"
    DASHBOARD_PORT: int = 8765
    DASHBOARD_CHART_BARS: int = 150


CFG = Config()

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("buy_only_bot")
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console output
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    # File output with forced flushing
    fh = logging.FileHandler(CFG.LOG_FILE)
    fh.setFormatter(fmt)
    # The 'delay=False' and flushing ensure we write immediately
    logger.addHandler(fh)
    
    return logger
log = setup_logger()
# ==============================================================================
# OPTION PRICING MODEL — used only by the backtester to model premiums
# ==============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_delta(S: float, K: float, T: float, sigma: float, r: float, option_type: str) -> Tuple[float, float]:
    """
    Black-Scholes European option price & delta.
    option_type: 'CE' (call) or 'PE' (put).
    Falls back to intrinsic value / a +-1 delta as T or sigma collapse to ~0,
    so this stays well-behaved right up to expiry instead of dividing by zero.
    """
    if T <= 1e-6 or sigma <= 0:
        if option_type == "CE":
            return max(S - K, 0.0), (1.0 if S > K else 0.0)
        return max(K - S, 0.0), (-1.0 if S < K else 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "CE":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    return max(price, 0.0), delta

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class OrderBlock:
    top: float
    bottom: float
    avg: float
    left_idx: int
    bullish: bool
    mitigated: bool = False

@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: int
    reason: str
    price: float

@dataclass
class OpenPosition:
    direction: int
    option_security_id: str
    option_symbol: str
    strike: int
    option_type: str
    entry_premium: float
    quantity: int
    highest_premium_seen: float
    stop_loss_premium: float
    target_premium: float
    entry_time: pd.Timestamp
    order_id: Optional[str] = None
    approx_delta: float = 0.5   # best-effort, from chain leg greeks if available

# ==============================================================================
# SIGNAL ENGINE & EMA LOGIC
# ==============================================================================

class StructureEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bull_obs: List[OrderBlock] = []
        self.bear_obs: List[OrderBlock] = []
        self.last_ce_dir: int = 0
        self.last_internal_trend: int = 0   

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def chandelier_exit(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        atr = self._atr(df, cfg.CE_ATR_PERIOD) * cfg.CE_ATR_MULT

        if cfg.CE_USE_CLOSE:
            highest = df["close"].rolling(cfg.CE_ATR_PERIOD).max()
            lowest = df["close"].rolling(cfg.CE_ATR_PERIOD).min()
        else:
            highest = df["high"].rolling(cfg.CE_ATR_PERIOD).max()
            lowest = df["low"].rolling(cfg.CE_ATR_PERIOD).min()

        long_stop = highest - atr
        short_stop = lowest + atr

        long_stop = long_stop.copy()
        short_stop = short_stop.copy()
        close = df["close"]

        for i in range(1, len(df)):
            prev_long = long_stop.iat[i - 1] if not np.isnan(long_stop.iat[i - 1]) else long_stop.iat[i]
            if close.iat[i - 1] > prev_long:
                long_stop.iat[i] = max(long_stop.iat[i], prev_long)
            prev_short = short_stop.iat[i - 1] if not np.isnan(short_stop.iat[i - 1]) else short_stop.iat[i]
            if close.iat[i - 1] < prev_short:
                short_stop.iat[i] = min(short_stop.iat[i], prev_short)

        direction = pd.Series(index=df.index, dtype="int64")
        d = 1
        for i in range(len(df)):
            prev_long = long_stop.iat[i - 1] if i > 0 else long_stop.iat[i]
            prev_short = short_stop.iat[i - 1] if i > 0 else short_stop.iat[i]
            if close.iat[i] > prev_short:
                d = 1
            elif close.iat[i] < prev_long:
                d = -1
            direction.iat[i] = d

        out = df.copy()
        out["ce_long_stop"] = long_stop
        out["ce_short_stop"] = short_stop
        out["ce_dir"] = direction
        return out

    @staticmethod
    def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
        n = len(series)
        result = pd.Series(False, index=series.index)
        for i in range(left + right, n):
            center = i - right
            window = series.iloc[center - left:i + 1]
            if len(window) == 0:
                continue
            if series.iat[center] == window.max() and (window == series.iat[center]).sum() == 1:
                result.iat[center] = True
        return result

    @staticmethod
    def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
        n = len(series)
        result = pd.Series(False, index=series.index)
        for i in range(left + right, n):
            center = i - right
            window = series.iloc[center - left:i + 1]
            if len(window) == 0:
                continue
            if series.iat[center] == window.min() and (window == series.iat[center]).sum() == 1:
                result.iat[center] = True
        return result

    def update_order_blocks(self, df: pd.DataFrame) -> None:
        cfg = self.cfg
        L = cfg.OB_LOOKBACK
        if len(df) < 2 * L + 2:
            return

        high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
        hl2 = (high + low) / 2
        upper = high.rolling(L).max()
        lower = low.rolling(L).min()

        if cfg.OB_FILTER_METHOD == "close":
            target_bull = close.rolling(L).min()
            target_bear = close.rolling(L).max()
        else:
            target_bull = lower
            target_bear = upper

        os_state = 0
        os_series = []
        for i in range(len(df)):
            lag = i - L
            if lag < 0:
                os_series.append(os_state)
                continue
            if high.iat[lag] > upper.iat[i] if i < len(upper) else False:
                os_state = 0
            elif low.iat[lag] < lower.iat[i] if i < len(lower) else False:
                os_state = 1
            os_series.append(os_state)
        os_arr = pd.Series(os_series, index=df.index)

        total_vol = vol.tail(100).sum()
        if total_vol == 0:
            ph_price = self._pivot_high(high, L, L)
            pl_price = self._pivot_low(low, L, L)
            phv = ph_price | pl_price 
        else:
            phv = self._pivot_high(vol, L, L)

        new_bull, new_bear = [], []
        n = len(df)
        for i in range(n):
            lag = i - L
            if lag < 0 or not phv.iat[i]:
                continue
            if os_arr.iat[lag] == 1:
                ob = OrderBlock(top=hl2.iat[lag], bottom=low.iat[lag], avg=low.iat[lag], left_idx=lag, bullish=True)
                new_bull.append(ob)
            elif os_arr.iat[lag] == 0:
                ob = OrderBlock(top=high.iat[lag], bottom=hl2.iat[lag], avg=high.iat[lag], left_idx=lag, bullish=False)
                new_bear.append(ob)

        last_target_bull = target_bull.iat[-1]
        last_target_bear = target_bear.iat[-1]
        new_bull = [ob for ob in new_bull if last_target_bull >= ob.bottom]
        new_bear = [ob for ob in new_bear if last_target_bear <= ob.top]

        self.bull_obs = new_bull[-cfg.MAX_ACTIVE_OBS:]
        self.bear_obs = new_bear[-cfg.MAX_ACTIVE_OBS:]

    def overlaps_bullish_ob(self, low: float, high: float) -> bool:
        return any(ob.bottom <= high and ob.top >= low for ob in self.bull_obs)

    def overlaps_bearish_ob(self, low: float, high: float) -> bool:
        return any(ob.bottom <= high and ob.top >= low for ob in self.bear_obs)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        df_ce = self.chandelier_exit(df)
        ce_dir_now = int(df_ce["ce_dir"].iat[-1])
        ce_dir_prev = int(df_ce["ce_dir"].iat[-2]) if len(df_ce) > 1 else ce_dir_now

        self.update_order_blocks(df)

        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        price = float(curr_bar["close"])
        ts = df.index[-1]

        ce_flip_long = ce_dir_prev == -1 and ce_dir_now == 1
        ce_flip_short = ce_dir_prev == 1 and ce_dir_now == -1

        ema3 = df['close'].ewm(span=3, adjust=False).mean()
        ema9 = df['close'].ewm(span=9, adjust=False).mean()
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        
        curr_ema3, prev_ema3 = ema3.iloc[-1], ema3.iloc[-2]
        curr_ema9, prev_ema9 = ema9.iloc[-1], ema9.iloc[-2]
        curr_ema21 = ema21.iloc[-1]

        candle_range = curr_bar['high'] - curr_bar['low']
        if candle_range == 0: candle_range = 1 
        bull_close_pct = (curr_bar['close'] - curr_bar['low']) / candle_range

        # EARLY SMC ANTICIPATION
        in_bull_ob = self.overlaps_bullish_ob(curr_bar['low'], curr_bar['high']) or self.overlaps_bullish_ob(prev_bar['low'], prev_bar['high'])
        in_bear_ob = self.overlaps_bearish_ob(curr_bar['low'], curr_bar['high']) or self.overlaps_bearish_ob(prev_bar['low'], prev_bar['high'])

        early_cross_up = curr_ema3 > curr_ema9 and prev_ema3 <= prev_ema9
        early_cross_down = curr_ema3 < curr_ema9 and prev_ema3 >= prev_ema9

        if early_cross_up and in_bull_ob and bull_close_pct >= 0.35:
            self.last_ce_dir = 1 
            return Signal(ts, 1, "Early Anticipation (Bull OB + 3/9 Cross)", price)

        if early_cross_down and in_bear_ob and bull_close_pct <= 0.65:
            self.last_ce_dir = -1 
            return Signal(ts, -1, "Early Anticipation (Bear OB + 3/9 Cross)", price)

        # SAFE BREAKOUT CATCH-UP
        if ce_flip_long:
            self.last_ce_dir = ce_dir_now
            trend_agrees = (price > curr_ema9) or (curr_ema9 >= curr_ema21)
            if trend_agrees and bull_close_pct >= 0.35:
                return Signal(ts, 1, "CE Bullish Breakout", price)
            else:
                log.info(f"Ignored Weak Buy: TrendAgrees={trend_agrees}, CloseStrength={bull_close_pct:.2f}")
                return Signal(ts, 0, "no signal", price)

        if ce_flip_short:
            self.last_ce_dir = ce_dir_now
            trend_agrees = (price < curr_ema9) or (curr_ema9 <= curr_ema21)
            if trend_agrees and bull_close_pct <= 0.65:
                return Signal(ts, -1, "CE Bearish Breakout", price)
            else:
                log.info(f"Ignored Weak Sell: TrendAgrees={trend_agrees}, CloseStrength={bull_close_pct:.2f}")
                return Signal(ts, 0, "no signal", price)

        self.last_ce_dir = ce_dir_now
        return Signal(ts, 0, "no signal", price)

    def current_ce_dir(self, df: pd.DataFrame) -> int:
        return int(self.chandelier_exit(df)["ce_dir"].iat[-1])

# ==============================================================================
# DHAN BROKER WRAPPER
# ==============================================================================

class DhanBroker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dhan = None
        if not cfg.DRY_RUN:
            self._connect()
        else:
            log.info("DRY_RUN=True -> paper trading mode active.")
            self._connect(quiet_on_fail=True)

    def _connect(self, quiet_on_fail: bool = False):
        try:
            from dhanhq import DhanContext, dhanhq
        except ImportError as e:
            msg = f"dhanhq library missing ({e}). Execute: pip install dhanhq"
            if quiet_on_fail:
                log.warning(msg)
                return
            raise RuntimeError(msg)

        if not self.cfg.DHAN_CLIENT_ID or not self.cfg.DHAN_ACCESS_TOKEN:
            msg = "Missing credentials. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables."
            if quiet_on_fail:
                log.warning(msg)
                return
            raise RuntimeError(msg)

        try:
            ctx = DhanContext(self.cfg.DHAN_CLIENT_ID, self.cfg.DHAN_ACCESS_TOKEN)
            self.dhan = dhanhq(ctx)
            self.dhan.get_positions()
            log.info("Connected to DhanHQ successfully.")
        except Exception as e:
            self.dhan = None
            msg = f"Dhan authentication error: {e}"
            if quiet_on_fail:
                log.warning(msg)
                return
            raise RuntimeError(msg)

    def get_intraday_candles(self, security_id: str, exchange_segment: str, instrument_type: str = "INDEX", days_back: int = 5) -> pd.DataFrame:
        if self.dhan is None:
            raise RuntimeError("Dhan connection down.")

        to_date = dt.datetime.now()
        from_date = to_date - dt.timedelta(days=days_back)
        resp = self.dhan.intraday_minute_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )

        if isinstance(resp, dict):
            status = str(resp.get("status", "")).lower()
            if status == "failure" or "errorMessage" in resp:
                return pd.DataFrame()
            data = resp.get("data", resp)
        else:
            data = resp

        if not isinstance(data, (dict, list)) or (isinstance(data, dict) and len(data) == 0):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        if "timestamp" not in df.columns and "start_Time" in df.columns:
            df = df.rename(columns={"start_Time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        ohlc_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in ohlc_cols if c in df.columns]].astype(float)

        tf = f"{self.cfg.TIMEFRAME_MINUTES}min"
        resampled = df.resample(tf).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        return resampled
        
    def get_historical_candles_chunked(self, security_id: str, exchange_segment: str, days: int) -> pd.DataFrame:
        if self.dhan is None:
            raise RuntimeError("Dhan connection down.")
        
        log.info(f"Downloading {days} days of historical data from Dhan. Please wait...")
        df_list = []
        end_dt = dt.datetime.now()
        start_dt = end_dt - dt.timedelta(days=days)
        
        current_end = end_dt
        while current_end > start_dt:
            current_start = max(current_end - dt.timedelta(days=25), start_dt)
            resp = self.dhan.intraday_minute_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type="INDEX",
                from_date=current_start.strftime("%Y-%m-%d"),
                to_date=current_end.strftime("%Y-%m-%d"),
            )
            
            if isinstance(resp, dict) and str(resp.get("status", "")).lower() == "success":
                data = resp.get("data", [])
                if data and len(data) > 0:
                    df_chunk = pd.DataFrame(data)
                    df_list.append(df_chunk)
            
            current_end = current_start - dt.timedelta(days=1)
            time.sleep(0.5) 
            
        if not df_list:
            return pd.DataFrame()
            
        df = pd.concat(df_list, ignore_index=True)
        if "timestamp" not in df.columns and "start_Time" in df.columns:
            df = df.rename(columns={"start_Time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        ohlc_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in ohlc_cols if c in df.columns]].astype(float)

        tf = f"{self.cfg.TIMEFRAME_MINUTES}min"
        resampled = df.resample(tf).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        return resampled

    def get_expired_option_candles(self, underlying_symbol: str, exchange_segment: str, interval: int,
                                    expiry_flag: str, expiry_code: int, strike: str, option_type: str,
                                    from_date: str, to_date: str) -> pd.DataFrame:
        """
        BEST-EFFORT helper, NOT wired into BacktestEngine by default.

        Dhan exposes a real "Expired/Rolling Options Data" endpoint
        (https://dhanhq.co/docs/v2/expired-options-data/) that returns
        actual historical OHLC + IV for options at a strike *relative to
        spot at that time* (e.g. "ATM", "ATM+1", "ATM-2", up to +/-10).
        That is the right way to backtest realistically — real traded data,
        not a model.

        It is not wired into BacktestEngine because this environment has no
        network access to api.dhan.co to verify the request/response shape
        end-to-end. Test this against a known date/strike yourself first.

        strike: "ATM", "ATM+1", "ATM-1", ... up to +/-10
        option_type: "CALL" or "PUT"
        interval: 1, 5, 15, 25, or 60 (minutes)
        Dhan limits this to ~30 days per call; this method chunks for you.
        """
        import requests  # local import: only needed if you actually use this helper

        if not self.cfg.DHAN_CLIENT_ID or not self.cfg.DHAN_ACCESS_TOKEN:
            raise RuntimeError("Missing Dhan credentials (set DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env vars).")

        url = "https://api.dhan.co/v2/charts/rollingoption"
        headers = {
            "Content-Type": "application/json",
            "access-token": self.cfg.DHAN_ACCESS_TOKEN,
            "client-id": self.cfg.DHAN_CLIENT_ID,
        }

        from_dt = dt.datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = dt.datetime.strptime(to_date, "%Y-%m-%d")

        frames = []
        chunk_start = from_dt
        while chunk_start < to_dt:
            chunk_end = min(chunk_start + dt.timedelta(days=29), to_dt)
            body = {
                "exchangeSegment": exchange_segment,
                "interval": str(interval),
                "securityId": int(self.cfg.UNDERLYING_SECURITY_ID.get(underlying_symbol, underlying_symbol)),
                "instrument": "OPTIDX",
                "expiryFlag": expiry_flag,
                "expiryCode": expiry_code,
                "strike": strike,
                "drvOptionType": option_type,
                "requiredData": ["open", "high", "low", "close", "volume"],
                "fromDate": chunk_start.strftime("%Y-%m-%d"),
                "toDate": chunk_end.strftime("%Y-%m-%d"),
            }
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            payload = data.get("data", data) if isinstance(data, dict) else data
            if payload:
                frames.append(pd.DataFrame(payload))
            chunk_start = chunk_end + dt.timedelta(days=1)
            time.sleep(0.5)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def get_option_chain(self, under_security_id: str, under_exchange_segment: str, expiry: str) -> dict:
        if self.dhan is None:
            raise RuntimeError("Dhan connection down.")
        return self.dhan.option_chain(
            under_security_id=int(under_security_id),
            under_exchange_segment=under_exchange_segment,
            expiry=expiry,
        )

    def get_nearest_expiry(self, under_security_id: str, under_exchange_segment: str) -> str:
        if self.dhan is None:
            raise RuntimeError("Dhan connection down.")
        resp = self.dhan.expiry_list(
            under_security_id=int(under_security_id), 
            under_exchange_segment=under_exchange_segment
        )
        dates = resp.get("data", resp) if isinstance(resp, dict) else resp
        if isinstance(dates, list) and dates:
            return sorted(dates)[0]
        raise RuntimeError(f"Could not retrieve stock expiry list: {resp}")

    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        if self.dhan is None:
            raise RuntimeError("Dhan connection down.")
        resp = self.dhan.ohlc_data(securities={exchange_segment: [int(security_id)]})
        try:
            data = resp["data"][exchange_segment][str(security_id)]
            return float(data["last_price"])
        except Exception:
            raise RuntimeError(f"Unexpected LTP payload: {resp}")

    def place_option_order(self, security_id: str, transaction_type: str, quantity: int, order_type: str = "MARKET", price: float = 0.0) -> dict:
        if self.cfg.DRY_RUN:
            sim_id = f"SIM-{int(time.time()*1000)}"
            log.info(f"[DRY RUN OPTION] {transaction_type}: security_id={security_id} qty={quantity} product=MARGIN")
            return {"orderId": sim_id, "orderStatus": "SIMULATED"}

        resp = self.dhan.place_order(
            security_id=security_id,
            exchange_segment=self.cfg.OPTION_EXCHANGE_SEGMENT,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type="MARGIN", 
            validity="DAY",
            price=price,
        )
        
        log.info(f"RAW DHAN RESPONSE: {resp}")

        if isinstance(resp, dict) and str(resp.get("status", "")).lower() == "failure":
            error_msg = resp.get("remarks") or resp.get("errorMessage") or resp
            log.error(f"DHAN ORDER REJECTED: {error_msg}")
            raise Exception(f"Broker rejected order: {error_msg}")
            
        log.info(f"LIVE OPTION EXECUTION RESP: {resp}")
        return resp

    def get_available_balance(self) -> float:
        if self.dhan is None:
            return self.cfg.CAPITAL
        try:
            resp = self.dhan.get_fund_limits()
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            return float(data.get("availabelBalance", data.get("availableBalance", self.cfg.CAPITAL)))
        except Exception:
            return self.cfg.CAPITAL

# ==============================================================================
# RISK MANAGER 
# ==============================================================================

class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.day_start_capital = cfg.CAPITAL
        self.realized_pnl_today: float = 0.0
        self.trades_today: int = 0
        self.trading_halted_today: bool = False
        self.current_date = dt.date.today()

    def new_day_if_needed(self):
        today = dt.date.today()
        if today != self.current_date:
            self.current_date = today
            self.realized_pnl_today = 0.0
            self.trades_today = 0
            self.trading_halted_today = False

    def record_trade_result(self, pnl: float):
        self.realized_pnl_today += pnl
        self.trades_today += 1
        
        cap = self.day_start_capital if self.day_start_capital > 0 else 100000
        max_loss = -(self.cfg.MAX_DAILY_LOSS_PCT / 100.0) * cap
        
        if self.realized_pnl_today <= max_loss:
            self.trading_halted_today = True

    def can_trade(self) -> Tuple[bool, str]:
        self.new_day_if_needed()
        if self.trading_halted_today:
            return False, "daily loss limit hit"
        if self.trades_today >= self.cfg.MAX_TRADES_PER_DAY:
            return False, "max trades reached"
        if dt.datetime.now().time() >= self.cfg.SQUARE_OFF_TIME:
            return False, "past market hours"
        return True, "ok"

# ==============================================================================
# TRADE LOGGER
# ==============================================================================

class TradeLogger:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["entry_time", "exit_time", "direction", "symbol", "strike", "option_type", "entry_premium", "exit_premium", "quantity", "pnl", "exit_reason"])

    def log(self, pos: OpenPosition, exit_time, exit_premium: float, pnl: float, exit_reason: str):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pos.entry_time, exit_time, "LONG" if pos.direction == 1 else "SHORT", pos.option_symbol, pos.strike, pos.option_type, pos.entry_premium, exit_premium, pos.quantity, pnl, exit_reason])

# ==============================================================================
# LIVE WEB DASHBOARD STATE
# ==============================================================================

class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "mode": "PAPER", "underlying": "", "last_update": None, "candles": [],          
            "bull_obs": [], "bear_obs": [], "signals": [], "position": None,       
            "risk": {"capital": 0, "pnl_today": 0, "trades_today": 0, "max_trades": 0, "halted": False, "max_daily_loss_pct": 0},
            "trades": [], "equity_curve": [], "status_message": "booting...",
        }

    def update(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)
            self._data["last_update"] = dt.datetime.now().isoformat()

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data, default=str))

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SMC+OB+CE — Live Dashboard</title>
<style>
  :root {
    --bg: #0a0e0f;
    --panel: #11161a;
    --line: #1d2429;
    --text: #e8edf0;
    --muted: #6f7c85;
    --green: #26a96c;
    --red: #e5484d;
    --amber: #d9a440;
    --blue: #4a90d9;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid var(--line);
  }
  .topbar h1 {
    font-size: 14px; font-weight: 600; letter-spacing: 0.04em;
    margin: 0; color: var(--text); text-transform: uppercase;
  }
  .topbar h1 span { color: var(--muted); font-weight: 400; }
  .badge {
    padding: 3px 10px; border-radius: 3px; font-size: 11px;
    letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;
  }
  .badge.paper { background: rgba(217,164,64,0.15); color: var(--amber); border: 1px solid rgba(217,164,64,0.4); }
  .badge.live { background: rgba(229,72,77,0.15); color: var(--red); border: 1px solid rgba(229,72,77,0.4); }

  .risk-strip {
    display: grid; grid-template-columns: repeat(5, 1fr);
    border-bottom: 1px solid var(--line);
  }
  .risk-cell {
    padding: 12px 20px; border-right: 1px solid var(--line);
  }
  .risk-cell:last-child { border-right: none; }
  .risk-label { color: var(--muted); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
  .risk-value { font-size: 18px; font-weight: 600; }
  .risk-value.pos { color: var(--green); }
  .risk-value.neg { color: var(--red); }
  .risk-value.halt { color: var(--red); }

  .layout { padding: 20px; display: grid; gap: 16px; grid-template-columns: 1fr 320px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .panel-header {
    padding: 10px 16px; border-bottom: 1px solid var(--line);
    color: var(--muted); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  }
  .panel-body { padding: 14px 16px; }

  canvas { display: block; width: 100%; }

  .position-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--line); }
  .position-row:last-child { border-bottom: none; }
  .position-row .k { color: var(--muted); }
  .empty-state { color: var(--muted); text-align: center; padding: 24px 0; font-size: 12px; }

  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  td { padding: 6px 8px; border-bottom: 1px solid var(--line); }
  tr:last-child td { border-bottom: none; }
  .dir-long { color: var(--green); }
  .dir-short { color: var(--red); }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }

  .footer-note { color: var(--muted); font-size: 11px; padding: 14px 20px; border-top: 1px solid var(--line); }
</style>
</head>
<body>
  <div class="topbar">
    <h1>SMC + OB + ChanEx <span id="underlying-name">/ —</span></h1>
    <span class="badge paper" id="mode-badge">PAPER</span>
  </div>

  <div class="risk-strip">
    <div class="risk-cell">
      <div class="risk-label">Capital</div>
      <div class="risk-value" id="r-capital">—</div>
    </div>
    <div class="risk-cell">
      <div class="risk-label">P&amp;L Today</div>
      <div class="risk-value" id="r-pnl">—</div>
    </div>
    <div class="risk-cell">
      <div class="risk-label">Trades Today</div>
      <div class="risk-value" id="r-trades">—</div>
    </div>
    <div class="risk-cell">
      <div class="risk-label">Daily Loss Cap</div>
      <div class="risk-value" id="r-cap">—</div>
    </div>
    <div class="risk-cell">
      <div class="risk-label">Status</div>
      <div class="risk-value" id="r-status">—</div>
    </div>
  </div>

  <div class="layout">
    <div style="display:flex; flex-direction:column; gap:16px;">
      <div class="panel">
        <div class="panel-header">Underlying — Order Blocks &amp; Signals</div>
        <div class="panel-body"><canvas id="chart" height="340"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-header">Equity Curve</div>
        <div class="panel-body"><canvas id="equity" height="140"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-header">Trade History</div>
        <div class="panel-body" style="padding:0;">
          <table id="trades-table">
            <thead>
              <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Strike</th><th>Qty</th><th>Entry ₹</th><th>Exit ₹</th><th>P&amp;L</th><th>Reason</th></tr>
            </thead>
            <tbody id="trades-body"></tbody>
          </table>
          <div class="empty-state" id="trades-empty">No closed trades yet.</div>
        </div>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:16px;">
      <div class="panel">
        <div class="panel-header">Open Position</div>
        <div class="panel-body" id="position-body">
          <div class="empty-state">No open position.</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">Recent Signals</div>
        <div class="panel-body" id="signals-body">
          <div class="empty-state">No signals yet.</div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer-note">
    Paper mode simulates orders only — figures here are not guaranteed to match live execution.
    Auto-refreshes every 3s. Last update: <span id="last-update">—</span>
  </div>

<script>
const fmt = (n, d=2) => (n === null || n === undefined || isNaN(n)) ? '—' : Number(n).toFixed(d);
const fmtMoney = (n) => (n === null || n === undefined || isNaN(n)) ? '—' : (n < 0 ? '-₹' : '₹') + Math.abs(n).toLocaleString('en-IN', {maximumFractionDigits: 0});

function drawChart(candles, bullObs, bearObs, signals) {
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth || 800, h = 340;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  if (!candles.length) {
    ctx.fillStyle = '#6f7c85'; ctx.font = '12px monospace';
    ctx.fillText('Waiting for candle data...', 16, h/2);
    return;
  }
  const pad = {l: 60, r: 16, t: 16, b: 24};
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const highs = candles.map(c => c.h), lows = candles.map(c => c.l);
  let maxP = Math.max(...highs), minP = Math.min(...lows);
  const obs = [...bullObs, ...bearObs];
  obs.forEach(o => { maxP = Math.max(maxP, o.top); minP = Math.min(minP, o.bottom); });
  const range = (maxP - minP) || 1;
  const yOf = (p) => pad.t + plotH - ((p - minP) / range) * plotH;
  const n = candles.length;
  const cw = plotW / n;

  ctx.strokeStyle = '#1d2429'; ctx.fillStyle = '#6f7c85'; ctx.font = '10px monospace';
  for (let i = 0; i <= 4; i++) {
    const p = minP + (range * i / 4);
    const y = yOf(p);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(p.toFixed(0), 4, y + 3);
  }

  bullObs.forEach(o => {
    ctx.fillStyle = 'rgba(38,169,108,0.12)';
    ctx.fillRect(pad.l, yOf(o.top), plotW, yOf(o.bottom) - yOf(o.top));
    ctx.strokeStyle = 'rgba(38,169,108,0.4)'; ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, yOf(o.top), plotW, yOf(o.bottom) - yOf(o.top));
  });
  bearObs.forEach(o => {
    ctx.fillStyle = 'rgba(229,72,77,0.12)';
    ctx.fillRect(pad.l, yOf(o.top), plotW, yOf(o.bottom) - yOf(o.top));
    ctx.strokeStyle = 'rgba(229,72,77,0.4)'; ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, yOf(o.top), plotW, yOf(o.bottom) - yOf(o.top));
  });

  candles.forEach((c, i) => {
    const x = pad.l + i * cw + cw / 2;
    const up = c.c >= c.o;
    ctx.strokeStyle = up ? '#26a96c' : '#e5484d';
    ctx.fillStyle = up ? '#26a96c' : '#e5484d';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, yOf(c.h)); ctx.lineTo(x, yOf(c.l)); ctx.stroke();
    const bodyTop = yOf(Math.max(c.o, c.c)), bodyBot = yOf(Math.min(c.o, c.c));
    ctx.fillRect(x - cw*0.35, bodyTop, cw*0.7, Math.max(bodyBot - bodyTop, 1));
  });

  signals.forEach(s => {
    const idx = candles.findIndex(c => c.t === s.t);
    if (idx === -1) return;
    const x = pad.l + idx * cw + cw / 2;
    const y = yOf(s.price);
    ctx.fillStyle = s.dir === 1 ? '#26a96c' : '#e5484d';
    ctx.beginPath();
    if (s.dir === 1) { ctx.moveTo(x, y+10); ctx.lineTo(x-5, y+18); ctx.lineTo(x+5, y+18); }
    else { ctx.moveTo(x, y-10); ctx.lineTo(x-5, y-18); ctx.lineTo(x+5, y-18); }
    ctx.closePath(); ctx.fill();
  });
}

function drawEquity(curve) {
  const canvas = document.getElementById('equity');
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth || 800, h = 140;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  if (!curve.length) {
    ctx.fillStyle = '#6f7c85'; ctx.font = '12px monospace';
    ctx.fillText('No equity data yet.', 16, h/2);
    return;
  }
  const pad = {l: 56, r: 16, t: 12, b: 16};
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const vals = curve.map(p => p.equity);
  const maxV = Math.max(...vals), minV = Math.min(...vals);
  const range = (maxV - minV) || 1;
  const yOf = (v) => pad.t + plotH - ((v - minV) / range) * plotH;
  const xOf = (i) => pad.l + (i / Math.max(curve.length - 1, 1)) * plotW;

  ctx.strokeStyle = '#1d2429';
  for (let i = 0; i <= 2; i++) {
    const v = minV + range * i / 2;
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#6f7c85'; ctx.font = '10px monospace';
    ctx.fillText(v.toFixed(0), 4, y + 3);
  }

  ctx.beginPath();
  curve.forEach((p, i) => {
    const x = xOf(i), y = yOf(p.equity);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#4a90d9'; ctx.lineWidth = 1.5; ctx.stroke();
}

async function refresh() {
  try {
    const res = await fetch('/api/state');
    const d = await res.json();

    document.getElementById('underlying-name').textContent = '/ ' + (d.underlying || '—');
    const badge = document.getElementById('mode-badge');
    badge.textContent = d.mode;
    badge.className = 'badge ' + (d.mode === 'LIVE' ? 'live' : 'paper');

    document.getElementById('r-capital').textContent = fmtMoney(d.risk.capital);
    const pnlEl = document.getElementById('r-pnl');
    pnlEl.textContent = fmtMoney(d.risk.pnl_today);
    pnlEl.className = 'risk-value ' + (d.risk.pnl_today >= 0 ? 'pos' : 'neg');
    document.getElementById('r-trades').textContent = `${d.risk.trades_today} / ${d.risk.max_trades}`;
    document.getElementById('r-cap').textContent = fmt(d.risk.max_daily_loss_pct, 1) + '%';
    const statusEl = document.getElementById('r-status');
    statusEl.textContent = d.risk.halted ? 'HALTED' : 'ACTIVE';
    statusEl.className = 'risk-value ' + (d.risk.halted ? 'halt' : 'pos');

    drawChart(d.candles, d.bull_obs, d.bear_obs, d.signals);
    drawEquity(d.equity_curve);

    const posBody = document.getElementById('position-body');
    if (d.position) {
      const p = d.position;
      posBody.innerHTML = `
        <div class="position-row"><span class="k">Direction</span><span class="${p.direction===1?'dir-long':'dir-short'}">${p.direction===1?'LONG (CE)':'SHORT (PE)'}</span></div>
        <div class="position-row"><span class="k">Strike</span><span>${p.strike} ${p.option_type}</span></div>
        <div class="position-row"><span class="k">Qty</span><span>${p.quantity}</span></div>
        <div class="position-row"><span class="k">Entry ₹</span><span>${fmt(p.entry_premium)}</span></div>
        <div class="position-row"><span class="k" style="color: #e5484d;">Trailing SL ₹</span><span style="color: #e5484d; font-weight: bold;">${fmt(p.stop_loss_premium)}</span></div>
        <div class="position-row"><span class="k" style="color: #26a96c;">Peak Price ₹</span><span style="color: #26a96c; font-weight: bold;">${fmt(p.target_premium)}</span></div>
        <div class="position-row"><span class="k">Entry Time</span><span>${(p.entry_time||'').toString().slice(0,19)}</span></div>
      `;
    } else {
      posBody.innerHTML = '<div class="empty-state">No open position.</div>';
    }

    const sigBody = document.getElementById('signals-body');
    if (d.signals && d.signals.length) {
      sigBody.innerHTML = d.signals.slice(-8).reverse().map(s => `
        <div class="position-row">
          <span class="k">${(s.t||'').toString().slice(5,16)}</span>
          <span class="${s.dir===1?'dir-long':'dir-short'}">${s.dir===1?'LONG':'SHORT'}</span>
        </div>
      `).join('');
    } else {
      sigBody.innerHTML = '<div class="empty-state">No signals yet.</div>';
    }

    const tbody = document.getElementById('trades-body');
    const emptyEl = document.getElementById('trades-empty');
    if (d.trades && d.trades.length) {
      emptyEl.style.display = 'none';
      tbody.innerHTML = d.trades.map(t => `
        <tr>
          <td>${(t.entry_time||'').toString().slice(5,16)}</td>
          <td>${(t.exit_time||'').toString().slice(5,16)}</td>
          <td class="${t.direction==='LONG'?'dir-long':'dir-short'}">${t.direction}</td>
          <td>${t.strike} ${t.option_type}</td>
          <td>${t.quantity}</td>
          <td>${fmt(t.entry_premium)}</td>
          <td>${fmt(t.exit_premium)}</td>
          <td class="${t.pnl>=0?'pnl-pos':'pnl-neg'}">${fmtMoney(t.pnl)}</td>
          <td>${t.exit_reason}</td>
        </tr>
      `).join('');
    } else {
      emptyEl.style.display = 'block';
      tbody.innerHTML = '';
    }

    document.getElementById('last-update').textContent = d.last_update ? new Date(d.last_update).toLocaleTimeString() : '—';
  } catch (e) {
    console.error('refresh failed', e);
  }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""

# ==============================================================================
# LIVE TRADING ENGINE
# ==============================================================================

class TradingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.broker = DhanBroker(cfg)
        self.engine = StructureEngine(cfg)
        self.risk = RiskManager(cfg)
        self.trade_log = TradeLogger(cfg.TRADE_LOG_CSV)
        self.open_position: Optional[OpenPosition] = None
        self.last_seen_bar: Optional[pd.Timestamp] = None
        self.dashboard_state = DashboardState() if cfg.DASHBOARD_ENABLED else None
        self.equity_curve: List[dict] = []
        self.recent_signals: List[dict] = []
        self._trade_history: List[dict] = []
        self.equity = cfg.CAPITAL

    def _underlying_ids(self) -> Tuple[str, str]:
        sec_id = self.cfg.UNDERLYING_SECURITY_ID[self.cfg.UNDERLYING]
        return sec_id, self.cfg.UNDERLYING_EXCHANGE_SEGMENT

    def _push_dashboard(self, df: Optional[pd.DataFrame] = None, status: str = ""):
        if self.dashboard_state is None: return
        cfg = self.cfg
        candles, bull_obs, bear_obs = [], [], []
        if df is not None and not df.empty:
            tail = df.tail(cfg.DASHBOARD_CHART_BARS)
            candles = [{"t": str(idx), "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close)} for idx, r in tail.iterrows()]
            bull_obs = [{"top": ob.top, "bottom": ob.bottom} for ob in self.engine.bull_obs]
            bear_obs = [{"top": ob.top, "bottom": ob.bottom} for ob in self.engine.bear_obs]

        position = None
        if self.open_position:
            p = self.open_position
            position = {"direction": p.direction, "strike": p.strike, "option_type": p.option_type, "quantity": p.quantity, "entry_premium": p.entry_premium, "stop_loss_premium": p.stop_loss_premium, "target_premium": p.highest_premium_seen, "entry_time": str(p.entry_time)}

        try: live_balance = self.broker.get_available_balance()
        except: live_balance = cfg.CAPITAL

        self.dashboard_state.update(
            mode="PAPER" if cfg.DRY_RUN else "LIVE", underlying=cfg.UNDERLYING,
            candles=candles, bull_obs=bull_obs, bear_obs=bear_obs, signals=self.recent_signals[-20:], position=position,
            risk={"capital": live_balance, "pnl_today": self.risk.realized_pnl_today, "trades_today": self.risk.trades_today, "max_trades": cfg.MAX_TRADES_PER_DAY, "halted": self.risk.trading_halted_today, "max_daily_loss_pct": cfg.MAX_DAILY_LOSS_PCT},
            trades=self._trade_history[-50:][::-1], equity_curve=self.equity_curve[-200:], status_message=status
        )

    def _select_and_enter(self, signal: Signal):
        cfg = self.cfg
        can, reason = self.risk.can_trade()
        if not can: return

        sec_id, seg = self._underlying_ids()
        try:
            expiry = self.broker.get_nearest_expiry(sec_id, seg)
            chain = self.broker.get_option_chain(sec_id, seg, expiry)
        except Exception as e:
            log.error(f"Option chain lookup breakdown: {e}")
            return

        try: available_balance = self.broker.get_available_balance()
        except Exception as e:
            log.error(f"Could not fetch available balance: {e}")
            return

        lot_size = cfg.LOT_SIZE.get(cfg.UNDERLYING, 65)
        option_type = "CE" if signal.direction == 1 else "PE"

        try: oc = chain["data"]["oc"]
        except KeyError: return

        # ... (inside _select_and_enter) ...
        
        affordable_options = []
        for strike_str, data in oc.items():
            try:
                strike_val = float(strike_str)
                leg_data = data.get(option_type.lower())
                
                if leg_data and "greeks" in leg_data:
                    # IDEA: Filter by Delta
                    delta = abs(float(leg_data["greeks"].get("delta", 0)))
                    
                    # Requirement: Delta must be 0.56 or higher
                    if delta >= 0.56:
                        premium = float(leg_data.get("last_price", 0))
                        if 0 < (premium * lot_size) <= available_balance:
                            affordable_options.append({"strike": strike_val, "premium": premium, "leg": leg_data})
            except Exception: continue

        # Now sort by the lowest delta that meets the criteria to keep it closest to ATM
        affordable_options.sort(key=lambda x: abs(float(x["leg"]["greeks"]["delta"])))
        best_choice = affordable_options[0]
        selected_strike, selected_leg, entry_premium = best_choice["strike"], best_choice["leg"], best_choice["premium"]

        option_security_id = str(selected_leg["security_id"])
        
        # STRICTLY 1 LOT
        quantity = lot_size 
        
        initial_trailing_sl = max(entry_premium - cfg.TRAILING_SL_POINTS, 0.05)

        # Best-effort: Dhan's option chain includes a "greeks" object with a
        # delta field when greeks data is available on the account/plan.
        # Used only for the optional ATR-based trailing stop; falls back to
        # 0.5 if absent or malformed so this never crashes order entry.
        approx_delta = 0.5
        try:
            greeks = selected_leg.get("greeks") or {}
            d_val = greeks.get("delta")
            if d_val is not None:
                approx_delta = float(d_val)
        except Exception:
            pass

        try: resp = self.broker.place_option_order(option_security_id, "BUY", quantity)
        except Exception as e: return

        self.open_position = OpenPosition(
            direction=signal.direction, option_security_id=option_security_id,
            option_symbol=selected_leg.get("tradingsymbol", f"{cfg.UNDERLYING}{selected_strike}{option_type}"),
            strike=selected_strike, option_type=option_type, entry_premium=entry_premium,
            quantity=quantity, stop_loss_premium=initial_trailing_sl, target_premium=float("inf"),
            highest_premium_seen=entry_premium, entry_time=signal.timestamp, order_id=resp.get("orderId"),
            approx_delta=approx_delta,
        )
        log.info(f"ENTERED OPTION: {self.open_position.option_symbol} Qty={quantity} (1 Lot) @ {entry_premium}")
        self._push_dashboard(status="position entered")

        # LOGGING THE FINAL TRADE ENTRY
        log.info(f"""
        --- TRADE ENTRY EXECUTED ---
        Symbol: {selected_leg.get('tradingsymbol')}
        Strike: {selected_strike}
        Type: {option_type}
        Entry Premium: {entry_premium}
        Delta: {selected_leg.get('greeks', {}).get('delta')}
        Theta: {selected_leg.get('greeks', {}).get('theta')}
        Gamma: {selected_leg.get('greeks', {}).get('gamma')}
        ----------------------------
        """)

    def _check_exit(self, df_underlying: pd.DataFrame):
        if self.open_position is None: return
        pos = self.open_position

        try: current_premium = self.broker.get_ltp(pos.option_security_id, self.cfg.OPTION_EXCHANGE_SEGMENT)
        except Exception: return

        if current_premium >= pos.entry_premium * 1.05:
            if pos.stop_loss_premium < pos.entry_premium:
                pos.stop_loss_premium = pos.entry_premium
                log.info(f"SL Shifted to CTC for {pos.option_symbol}")

        if current_premium > pos.highest_premium_seen:
            pos.highest_premium_seen = current_premium

        if self.cfg.USE_ATR_TRAILING_SL:
            try:
                atr_now = float(self.engine._atr(df_underlying, self.cfg.CE_ATR_PERIOD).iat[-1])
                buffer = max(abs(pos.approx_delta) * atr_now * self.cfg.ATR_TRAIL_MULT, 1.0)
                trailing_sl_price = max(pos.highest_premium_seen - buffer, 0.05)
            except Exception:
                trailing_sl_price = max(pos.highest_premium_seen - self.cfg.TRAILING_SL_POINTS, 0.05)
        else:
            trailing_sl_price = max(pos.highest_premium_seen - self.cfg.TRAILING_SL_POINTS, 0.05)

        pos.stop_loss_premium = trailing_sl_price 

        exit_reason = None
        
        # 1. Check Trailing SL
        if current_premium <= trailing_sl_price:
            exit_reason = f"trailing_stop_loss (Peak was {pos.highest_premium_seen:.2f})"
        
        # 2. Check strict Chandelier Exit Reversal
        elif self.cfg.TRAIL_WITH_CHANDELIER:
            ce_dir = self.engine.current_ce_dir(df_underlying)
            if (pos.direction == 1 and ce_dir == -1) or (pos.direction == -1 and ce_dir == 1):
                exit_reason = "chandelier_flip"

        # 3. Market close
        if dt.datetime.now().time() >= self.cfg.SQUARE_OFF_TIME:
            exit_reason = exit_reason or "square_off"

        if exit_reason:
            self._exit_position(pos, current_premium, exit_reason)

    def _exit_position(self, pos: OpenPosition, exit_premium: float, reason: str):
        self.broker.place_option_order(pos.option_security_id, "SELL", pos.quantity)
        pnl = (exit_premium - pos.entry_premium) * pos.quantity
        self.risk.record_trade_result(pnl)
        self.trade_log.log(pos, dt.datetime.now(), exit_premium, pnl, reason)
        log.info(f"EXITED OPTION: {pos.option_symbol} @ {exit_premium} | Reason={reason} | PnL={pnl:.2f}")

        self.equity += pnl
        now_iso = dt.datetime.now().isoformat()
        self._trade_history.append({"entry_time": str(pos.entry_time), "exit_time": now_iso, "direction": "LONG" if pos.direction == 1 else "SHORT", "strike": pos.strike, "option_type": pos.option_type, "quantity": pos.quantity, "entry_premium": pos.entry_premium, "exit_premium": exit_premium, "pnl": pnl, "exit_reason": reason})
        self.equity_curve.append({"t": now_iso, "equity": self.equity})
        self.open_position = None

    def run_forever(self):
        cfg = self.cfg
        log.info(f"Booting Option Automation Engine | Target: {cfg.UNDERLYING}")

        if self.broker.dhan is None:
            log.error("Broker pipeline offline.")
            return

        sec_id, seg = self._underlying_ids()
        
        if cfg.DASHBOARD_ENABLED:
            from flask import Flask, jsonify, Response
            app = Flask(__name__)
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            @app.route("/")
            def index(): return Response(DASHBOARD_HTML, mimetype="text/html")
            @app.route("/api/state")
            def api_state(): return jsonify(self.dashboard_state.snapshot())
            threading.Thread(target=lambda: app.run(host=cfg.DASHBOARD_HOST, port=cfg.DASHBOARD_PORT, debug=True, use_reloader=False), daemon=True).start()

        while True:
            try:
                log.info("Loop Heartbeat: Checking for new candles...")
                
                df = self.broker.get_intraday_candles(sec_id, seg, "INDEX")
                if df.empty or len(df) < max(cfg.SWING_LENGTH, cfg.CE_ATR_PERIOD) + 5:
                    time.sleep(cfg.POLL_SECONDS)
                    continue

                latest_bar_time = df.index[-1]
                
                if self.open_position is not None:
                    self._check_exit(df)

                if latest_bar_time != self.last_seen_bar:
                    self.last_seen_bar = latest_bar_time
                    df_closed = df.iloc[:-1] 
                    
                    if not df_closed.empty and len(df_closed) > 5:
                        signal = self.engine.generate_signal(df_closed)
                        
                        if signal.direction != 0:
                            self.recent_signals.append({"t": str(signal.timestamp), "price": signal.price, "dir": signal.direction, "reason": signal.reason})
                            if self.open_position is None:
                                self._select_and_enter(signal)

                self._push_dashboard(df, status="scanning panels")
                time.sleep(cfg.POLL_SECONDS)

            except KeyboardInterrupt: break
            except Exception as e:
                time.sleep(cfg.POLL_SECONDS)

# ==============================================================================
# PROFESSIONAL BACKTESTING ENGINE (1 Lot Strict)
# ==============================================================================

class BacktestEngine:
    def __init__(self, cfg: Config, days: int):
        self.cfg = cfg
        self.days = days
        self.broker = DhanBroker(cfg)
        self.engine = StructureEngine(cfg)
        
        self.initial_capital = cfg.CAPITAL
        self.capital = cfg.CAPITAL
        self.peak_capital = cfg.CAPITAL
        self.max_drawdown_pct = 0.0
        
        self.open_pos = None
        self.trade_history = []
        
        # Dhan Brokerage + STT + Stamp Duty + Exchange Charges (~₹60 per complete F&O trade)
        self.brokerage_per_trade = 60.0 

    # def _select_strike_and_premium(self, S: float, T: float, sigma: float, r: float, option_type: str,
    #                                 lot_size: int, capital: float, strike_step: int) -> Optional[Tuple[float, float]]:
    #     """
    #     Mirrors the LIVE bot's strike selection: it scans the whole chain and
    #     picks whichever strike has the highest premium that still costs
    #     <= available capital for 1 lot. Here we scan a strike grid around
    #     spot and price each one with Black-Scholes to find the same thing.
    #     """
    #     budget_premium = capital / lot_size
    #     atm = round(S / strike_step) * strike_step
    #     best = None
    #     for n in range(-self.cfg.BACKTEST_STRIKE_SCAN_STEPS, self.cfg.BACKTEST_STRIKE_SCAN_STEPS + 1):
    #         K = atm + n * strike_step
    #         if K <= 0:
    #             continue
    #         price, _ = bs_price_delta(S, K, T, sigma, r, option_type)
    #         if 0 < price <= budget_premium:
    #             if best is None or price > best[1]:
    #                 best = (K, price)
    #     return best
    # Instead of current logic, use this filtered approach:
    def _select_strike_and_premium(self, S, T, sigma, r, option_type, lot_size, capital, strike_step):
        atm_strike = round(S / strike_step) * strike_step
        
        # "1 Level ITM" logic:
        # If CE, 1 Level ITM is (ATM - StrikeStep). If PE, 1 Level ITM is (ATM + StrikeStep)
        target_strike = (atm_strike - strike_step) if option_type == "CE" else (atm_strike + strike_step)
        
        # Calculate price and delta for this specific strike
        price, delta = bs_price_delta(S, target_strike, T, sigma, r, option_type)
        
        # Validate against Delta > 0.56 requirement
        if abs(delta) >= 0.56:
            return target_strike, price
        return None # If it doesn't meet criteria, don't trade

    def run(self):
        log.info(f"\n{'='*60}\nINITIALIZING STRICT 1-LOT BACKTEST ({self.days} DAYS)\n{'='*60}")
        sec_id, seg = self.cfg.UNDERLYING_SECURITY_ID[self.cfg.UNDERLYING], self.cfg.UNDERLYING_EXCHANGE_SEGMENT
        
        df = self.broker.get_historical_candles_chunked(sec_id, seg, self.days)
        if df.empty:
            log.error("Failed to fetch historical data for backtesting.")
            return

        log.info(f"Loaded {len(df)} historical 5-minute candles. Simulating strategy...")
        log.info(
            f"Premiums are MODELED via Black-Scholes (IV={self.cfg.BACKTEST_IV_ASSUMPTION:.0%}, "
            f"r={self.cfg.BACKTEST_RISK_FREE_RATE:.0%}, assumed days-to-expiry="
            f"{self.cfg.BACKTEST_ASSUMED_DAYS_TO_EXPIRY}) — an approximation, not historical tick data. "
            f"See DhanBroker.get_expired_option_candles() for a path to real historical option data."
        )

        if len(self.trade_history) == 0 and self.days < 30:
            log.info(
                f"NOTE: {self.days}-day backtest will likely produce too few trades to draw any "
                f"conclusion from. Consider --backtest 60 or higher before judging win rate."
            )

        lot_size = self.cfg.LOT_SIZE.get(self.cfg.UNDERLYING, 65)
        strike_step = self.cfg.STRIKE_STEP.get(self.cfg.UNDERLYING, 50)
        sigma = self.cfg.BACKTEST_IV_ASSUMPTION
        r = self.cfg.BACKTEST_RISK_FREE_RATE

        minutes_per_trading_day = 375.0   # NSE index session, 09:15-15:30
        trading_days_per_year = 252.0
        T_entry_years = self.cfg.BACKTEST_ASSUMED_DAYS_TO_EXPIRY / trading_days_per_year
        bar_minutes = self.cfg.TIMEFRAME_MINUTES

        for i in range(50, len(df)):
            df_slice = df.iloc[:i]
            curr_bar = df_slice.iloc[-1]
            idx_time = df_slice.index[-1]

            if self.open_pos is not None:
                pos = self.open_pos

                bars_elapsed = i - pos['entry_bar_idx']
                minutes_elapsed = bars_elapsed * bar_minutes
                T_remaining = max(
                    pos['T_entry'] - (minutes_elapsed / (trading_days_per_year * minutes_per_trading_day)),
                    1e-6,
                )

                # REALISTIC WICK DETECTION: use the bar's high/low as the
                # best/worst underlying price the position could have seen.
                if pos['direction'] == 1:
                    worst_S, best_S = curr_bar['low'], curr_bar['high']
                else:
                    worst_S, best_S = curr_bar['high'], curr_bar['low']

                best_sim_premium, _ = bs_price_delta(best_S, pos['strike'], T_remaining, sigma, r, pos['option_type'])
                worst_sim_premium, _ = bs_price_delta(worst_S, pos['strike'], T_remaining, sigma, r, pos['option_type'])

                if best_sim_premium > pos['highest_premium_seen']:
                    pos['highest_premium_seen'] = best_sim_premium

                if self.cfg.USE_ATR_TRAILING_SL:
                    atr_now = float(self.engine._atr(df_slice, self.cfg.CE_ATR_PERIOD).iat[-1])
                    _, delta_now = bs_price_delta(curr_bar['close'], pos['strike'], T_remaining, sigma, r, pos['option_type'])
                    buffer = max(abs(delta_now) * atr_now * self.cfg.ATR_TRAIL_MULT, 1.0)
                    trailing_sl = max(pos['highest_premium_seen'] - buffer, 0.05)
                else:
                    trailing_sl = max(pos['highest_premium_seen'] - self.cfg.TRAILING_SL_POINTS, 0.05)

                exit_reason = None
                exit_premium = 0.0

                # 1. Did the WORST price of the candle hit our trailing stop?
                if worst_sim_premium <= trailing_sl:
                    exit_reason = f"trailing_sl_hit (Peak: {pos['highest_premium_seen']:.2f})"
                    exit_premium = trailing_sl

                # 2. Strict Chandelier Exit flip (checked at candle close)
                if exit_reason is None and self.cfg.TRAIL_WITH_CHANDELIER:
                    ce_dir = self.engine.current_ce_dir(df_slice)
                    if (pos['direction'] == 1 and ce_dir == -1) or (pos['direction'] == -1 and ce_dir == 1):
                        exit_reason = "chandelier_flip"
                        exit_premium, _ = bs_price_delta(curr_bar['close'], pos['strike'], T_remaining, sigma, r, pos['option_type'])

                # 3. Market close square off
                if idx_time.time() >= self.cfg.SQUARE_OFF_TIME and exit_reason is None:
                    exit_reason = "square_off"
                    exit_premium, _ = bs_price_delta(curr_bar['close'], pos['strike'], T_remaining, sigma, r, pos['option_type'])

                if exit_reason:
                    gross_pnl = (exit_premium - pos['entry_premium']) * pos['quantity']
                    net_pnl = gross_pnl - self.brokerage_per_trade

                    self.capital += net_pnl
                    self.peak_capital = max(self.peak_capital, self.capital)
                    dd = (self.peak_capital - self.capital) / self.peak_capital * 100
                    self.max_drawdown_pct = max(self.max_drawdown_pct, dd)

                    self.trade_history.append({
                        "entry_time": pos['entry_time'], "exit_time": idx_time,
                        "dir": "LONG" if pos['direction'] == 1 else "SHORT",
                        "qty": pos['quantity'], "net_pnl": net_pnl, "reason": exit_reason,
                        "strike": pos['strike'], "option_type": pos['option_type'],
                        "entry_premium": pos['entry_premium'], "exit_premium": exit_premium,
                    })
                    self.open_pos = None

            else:
                df_closed = df_slice.iloc[:-1]
                if len(df_closed) > 5:
                    signal = self.engine.generate_signal(df_closed)

                    if signal.direction != 0 and idx_time.time() < self.cfg.SQUARE_OFF_TIME:
                        option_type = "CE" if signal.direction == 1 else "PE"
                        selection = self._select_strike_and_premium(
                            S=signal.price, T=T_entry_years, sigma=sigma, r=r,
                            option_type=option_type, lot_size=lot_size,
                            capital=self.capital, strike_step=strike_step,
                        )
                        if selection is not None:
                            strike, entry_premium = selection
                            self.open_pos = {
                                "direction": signal.direction,
                                "entry_idx_price": signal.price,
                                "strike": strike,
                                "option_type": option_type,
                                "entry_premium": entry_premium,
                                "quantity": lot_size,
                                "highest_premium_seen": entry_premium,
                                "entry_time": signal.timestamp,
                                "entry_bar_idx": i,
                                "T_entry": T_entry_years,
                            }

        self._print_report()

    def _print_report(self):
        trades = self.trade_history
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]

        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        total_pnl = sum(t['net_pnl'] for t in trades)
        gross_profit = sum(t['net_pnl'] for t in wins)
        gross_loss = abs(sum(t['net_pnl'] for t in losses))
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        expectancy = (total_pnl / len(trades)) if trades else 0.0

        print(f"\n{'='*60}")
        print(f" 1-LOT BACKTEST REPORT: {self.cfg.UNDERLYING} ({self.days} Days)")
        print(f"{'='*60}")
        print(f" Total Trades Taken : {len(trades)}")
        print(f" Win Rate           : {win_rate:.2f}%")
        print(f" Total Wins / Losses: {len(wins)} / {len(losses)}")
        print(f" Avg Win / Avg Loss : ₹{avg_win:,.2f} / ₹{avg_loss:,.2f}")
        print(f" Profit Factor      : {profit_factor:.2f}")
        print(f" Expectancy / Trade : ₹{expectancy:,.2f}")
        print(f" Max Drawdown       : {self.max_drawdown_pct:.2f}%")
        print(f" Brokerage Paid     : ₹{len(trades) * self.brokerage_per_trade:,.2f}")
        print(f" Starting Capital   : ₹{self.initial_capital:,.2f}")
        print(f" Final Capital      : ₹{self.capital:,.2f}")
        print(f" NET PROFIT (PnL)   : {'+' if total_pnl > 0 else ''}₹{total_pnl:,.2f}")
        print(f"{'='*60}")
        if len(trades) < 30:
            print(f" NOTE: only {len(trades)} trade(s) in this sample — not enough to draw any")
            print(f" statistical conclusion, good or bad. Re-run with more --backtest days.")
            print(f"{'='*60}")
        print()

# ==============================================================================
# MAIN PARSER ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                    help="Required to place real orders. Without this flag, the bot always runs in paper/dry-run mode.")
    p.add_argument("--backtest", type=int, default=0, help="Number of days to backtest (e.g. 90)")
    args = p.parse_args()

    if args.backtest > 0:
        CFG.DRY_RUN = True
        CFG.DASHBOARD_ENABLED = False
        bt = BacktestEngine(CFG, days=args.backtest)
        bt.run()

    elif args.live:
        CFG.DRY_RUN = False
        log.warning("!!! --live PASSED: THIS WILL TRADE REAL NIFTY OPTIONS WITH REAL MONEY !!!")
        confirm = input("Type EXACTLY 'YES I UNDERSTAND' to proceed live, anything else aborts: ")
        if confirm.strip() != "YES I UNDERSTAND":
            log.info("Live trading not confirmed. Exiting without placing any orders.")
            sys.exit(0)
        time.sleep(3)
        engine = TradingEngine(CFG)
        engine.run_forever()

    else:
        CFG.DRY_RUN = True
        log.info("No --live flag passed -> running in PAPER (dry-run) mode. Pass --live to trade with real money.")
        engine = TradingEngine(CFG)
        engine.run_forever()