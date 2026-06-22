#!/usr/bin/env python3
"""
================================================================================
 SMC + Order Block + Chandelier Exit — Intraday Options Bot for DhanHQ
================================================================================
Strategy logic (adapted from the LuxAlgo "SMC+OB+chanEX" Pine Script indicator
you provided) re-implemented in Python and wired to DhanHQ v2 for execution:

  1. Swing structure (BOS / CHoCH) on the underlying index (NIFTY/BANKNIFTY)
  2. Internal order blocks (bullish/bearish), detected from pivot volume highs
  3. Chandelier Exit (ATR trailing stop) used as the trend filter + trail-stop
  4. Entry rule (long example):
         - Chandelier Exit flips bullish (dir changes -1 -> +1)
         AND
         - Price is at/above an unmitigated bullish internal order block
            (or price broke structure bullish - BOS/CHoCH internal)
     Mirrored for shorts.
  5. On a long signal -> BUY ATM/near-ATM CE option.
     On a short signal -> BUY ATM/near-ATM PE option.
     (We trade options by buying premium in the direction of the signal -
      we do NOT trade the option's own price action for structure, because
      premium time-decay and IV noise make OB/structure on the option chart
      meaningless. Signals always come from the underlying.)

--------------------------------------------------------------------------------
 IMPORTANT / READ BEFORE USE
--------------------------------------------------------------------------------
 - NO strategy can be guaranteed to produce an 80-90% win rate or sub-10%
   drawdown in live markets. Those numbers depend entirely on market regime,
   parameters, instrument, and luck. This script gives you the tools to
   MEASURE win rate / drawdown on your own historical data via --backtest,
   and to trade live/paper with strict risk controls - it does not and
   cannot guarantee those targets.
 - DEFAULT MODE IS PAPER TRADING (DRY_RUN = True). No real orders are sent
   to Dhan until you explicitly set DRY_RUN = False (or pass --live).
 - This is a single, self-contained Python file as requested. Drop it into
   your git repo and run it directly.
 - You are solely responsible for testing this thoroughly in paper mode and
   for all consequences of running it with real money.

--------------------------------------------------------------------------------
 SETUP
--------------------------------------------------------------------------------
 pip install dhanhq pandas numpy flask

 Set these in your environment (recommended) or directly in CONFIG below:
   export DHAN_CLIENT_ID="your_client_id"
   export DHAN_ACCESS_TOKEN="your_access_token"

--------------------------------------------------------------------------------
 WEB DASHBOARD
--------------------------------------------------------------------------------
 A live dashboard runs automatically alongside the trading engine (same
 process, separate thread) at:
     http://127.0.0.1:8765
 It shows the live underlying chart with order-block zones + signal markers,
 current position, risk status, trade history, and equity curve. Disable it
 with --no-dashboard if you don't want it.

--------------------------------------------------------------------------------
 USAGE
--------------------------------------------------------------------------------
 Paper-trade live market data (no real orders), default mode:
   python algo_bot.py

 Go live (real orders) - only after you've validated in paper mode:
   python algo_bot.py --live

 Backtest on historical data:
   python algo_bot.py --backtest --underlying NIFTY --from 2024-01-01 --to 2024-06-01

================================================================================
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import logging
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
    # MUST be flipped to False explicitly (or --live passed) to place real
    # orders. Defaults to paper trading.
    DRY_RUN: bool = True

    # ---- DhanHQ credentials --------------------------------------------
    DHAN_CLIENT_ID: str = "1110569990"
    DHAN_ACCESS_TOKEN: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgyMTg2MzIwLCJpYXQiOjE3ODIwOTk5MjAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTEwNTY5OTkwIn0.LHn9fx3rowd5wq6TRxvA4wT4y-jqD_4_NtvIFbPR1rmARvFHP0EsTkRsATEihPAGm_xPcXjh09xLL73udEiH7Q"

    # ---- Instrument ------------------------------------------------------
    UNDERLYING: str = "NIFTY"          # "NIFTY" or "BANKNIFTY"
    # Dhan security IDs for the index itself (for fetching spot/candle data).
    # 13 = NIFTY 50 index, 25 = NIFTY BANK index (IDX_I segment). Verify these
    # against dhan.fetch_security_list("compact") before going live - Dhan
    # occasionally revises IDs.
    UNDERLYING_SECURITY_ID: Dict[str, str] = {"NIFTY": "13", "BANKNIFTY": "25"}
    UNDERLYING_EXCHANGE_SEGMENT: str = "IDX_I"
    OPTION_EXCHANGE_SEGMENT: str = "NSE_FNO"

    # ---- Candle timeframe --------------------------------------------
    TIMEFRAME_MINUTES: int = 5         # 5 or 15 recommended for intraday
    POLL_SECONDS: int = 20             # how often to poll for a new candle

    # ---- Strategy parameters (mirrors the Pine Script inputs) ----------
    SWING_LENGTH: int = 50             # swing structure lookback
    INTERNAL_LENGTH: int = 5           # internal structure lookback
    OB_LOOKBACK: int = 5               # order block pivot lookback (lengthOB)
    OB_FILTER_METHOD: str = "atr"      # "atr" or "range"
    OB_MITIGATION: str = "highlow"     # "close" or "highlow"
    MAX_ACTIVE_OBS: int = 5            # how many internal OBs to track

    CE_ATR_PERIOD: int = 22            # Chandelier Exit ATR period
    CE_ATR_MULT: float = 3.0           # Chandelier Exit ATR multiplier
    CE_USE_CLOSE: bool = True          # use close for extremums (vs high/low)

    # ---- Risk management (this is what keeps drawdown bounded) ---------
    RISK_PER_TRADE_PCT: float = 1.0      # % of capital risked per trade
    CAPITAL: float = 100000.0            # trading capital, used for sizing
    MAX_DAILY_LOSS_PCT: float = 3.0      # circuit breaker: stop for the day
    MAX_TRADES_PER_DAY: int = 4          # hard cap on number of trades/day
    STOP_LOSS_PCT_OF_PREMIUM: float = 25.0   # SL = 25% below entry premium
    TARGET_PCT_OF_PREMIUM: float = 50.0      # optional fixed target (0 disables)
    TRAIL_WITH_CHANDELIER: bool = True       # also exit on CE flip against position
    MAX_LOTS_PER_TRADE: int = 10             # absolute cap regardless of sizing
    SQUARE_OFF_TIME: dt.time = dt.time(15, 15)  # force-exit all positions

    # Lot sizes (verify current values on NSE before trading - these change
    # periodically; do not rely on this script for live exchange specs).
    LOT_SIZE: Dict[str, int] = {"NIFTY": 75, "BANKNIFTY": 35}

    # Strike selection
    STRIKE_SELECTION: str = "ATM"      # "ATM" or "ATM+1" / "ATM-1" style offset
    STRIKE_STEP: Dict[str, int] = {"NIFTY": 50, "BANKNIFTY": 100}

    # ---- Logging / persistence ------------------------------------------
    LOG_FILE: str = "algo_bot.log"
    TRADE_LOG_CSV: str = "trades_log.csv"
    STATE_FILE: str = "bot_state.json"

    # ---- Web dashboard ----------------------------------------------------
    DASHBOARD_ENABLED: bool = True
    DASHBOARD_HOST: str = "127.0.0.1"
    DASHBOARD_PORT: int = 8765
    DASHBOARD_CHART_BARS: int = 150   # how many recent candles to plot


CFG = Config()

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("smc_ob_ce_bot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(CFG.LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


log = setup_logger()

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
    direction: int          # +1 long, -1 short, 0 none
    reason: str
    price: float


@dataclass
class OpenPosition:
    direction: int                  # +1 long (bought CE), -1 short (bought PE)
    option_security_id: str
    option_symbol: str
    strike: int
    option_type: str               # "CE" or "PE"
    entry_premium: float
    quantity: int
    stop_loss_premium: float
    target_premium: float
    entry_time: pd.Timestamp
    order_id: Optional[str] = None


# ==============================================================================
# SIGNAL ENGINE  -- Swing/Internal Structure + Order Blocks + Chandelier Exit
#
# This is a faithful, simplified-for-Python re-implementation of the relevant
# parts of the LuxAlgo Pine Script you supplied: pivot-based swing detection,
# BOS/CHoCH classification, internal order block creation from pivot-volume
# bars, order block mitigation, and the Chandelier Exit trend filter.
# ==============================================================================

class StructureEngine:
    """
    Maintains rolling swing/internal structure state and order blocks for one
    OHLCV stream (the underlying index), and produces entry signals by
    combining: (a) most recent BOS/CHoCH direction, (b) live, unmitigated
    internal order blocks, (c) Chandelier Exit direction flip.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bull_obs: List[OrderBlock] = []
        self.bear_obs: List[OrderBlock] = []
        self.last_ce_dir: int = 0
        self.last_internal_trend: int = 0   # +1 bullish, -1 bearish

    # ---------------- Chandelier Exit -------------------------------------
    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
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

        # Recursive ratchet exactly mirrors the Pine Script logic:
        #   longStop := close[1] > longStopPrev ? max(longStop, longStopPrev) : longStop
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

    # ---------------- Pivot helpers ---------------------------------------
    @staticmethod
    def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
        """True at index i if series[i-left] is the highest in [i-left-left, i-left+right]."""
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

    # ---------------- Internal structure (BOS/CHoCH) -----------------------
    def detect_internal_trend(self, df: pd.DataFrame) -> int:
        """
        Simplified internal market structure: compares the most recent
        internal swing high/low break to classify trend as +1 / -1 / 0.
        Uses INTERNAL_LENGTH as the pivot lookback (mirrors `internal` swings
        in the original script).
        """
        L = self.cfg.INTERNAL_LENGTH
        if len(df) < 2 * L + 2:
            return self.last_internal_trend

        ph = self._pivot_high(df["high"], L, L)
        pl = self._pivot_low(df["low"], L, L)

        last_swing_high = None
        last_swing_low = None
        trend = self.last_internal_trend
        closes = df["close"].values

        for i in range(len(df)):
            if ph.iat[i]:
                last_swing_high = df["high"].iat[i]
            if pl.iat[i]:
                last_swing_low = df["low"].iat[i]
            c = closes[i]
            if last_swing_high is not None and c > last_swing_high:
                trend = 1
                last_swing_high = None  # consumed -> wait for next pivot (BOS)
            if last_swing_low is not None and c < last_swing_low:
                trend = -1
                last_swing_low = None

        self.last_internal_trend = trend
        return trend

    # ---------------- Internal Order Blocks --------------------------------
    def update_order_blocks(self, df: pd.DataFrame) -> None:
        """
        Re-derive internal order blocks from pivot-volume bars, mirroring
        get_coordinates()/remove_mitigated() in the Pine source, restricted
        to internal-length pivots (order block lookback = OB_LOOKBACK).
        """
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

        # os[] state: 0 -> bearish-biased zone, 1 -> bullish-biased zone
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

        phv = self._pivot_high(vol, L, L)

        # Build/refresh OB lists from scratch each call (bounded by recent
        # window for performance - fine for an intraday bot with rolling data)
        new_bull, new_bear = [], []
        n = len(df)
        for i in range(n):
            lag = i - L
            if lag < 0 or not phv.iat[i]:
                continue
            if os_arr.iat[lag] == 1:
                ob = OrderBlock(top=hl2.iat[lag], bottom=low.iat[lag],
                                 avg=low.iat[lag], left_idx=lag, bullish=True)
                new_bull.append(ob)
            elif os_arr.iat[lag] == 0:
                ob = OrderBlock(top=high.iat[lag], bottom=hl2.iat[lag],
                                 avg=high.iat[lag], left_idx=lag, bullish=False)
                new_bear.append(ob)

        # Mitigation: remove OBs whose level has since been breached
        last_target_bull = target_bull.iat[-1]
        last_target_bear = target_bear.iat[-1]
        new_bull = [ob for ob in new_bull if last_target_bull >= ob.bottom]
        new_bear = [ob for ob in new_bear if last_target_bear <= ob.top]

        # Keep most recent N
        self.bull_obs = new_bull[-cfg.MAX_ACTIVE_OBS:]
        self.bear_obs = new_bear[-cfg.MAX_ACTIVE_OBS:]

    def price_in_bullish_ob(self, price: float) -> bool:
        return any(ob.bottom <= price <= ob.top for ob in self.bull_obs)

    def price_in_bearish_ob(self, price: float) -> bool:
        return any(ob.bottom <= price <= ob.top for ob in self.bear_obs)

    # ---------------- Combined signal ---------------------------------------
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Long signal when:
          - Chandelier Exit just flipped to +1 (bullish), AND
          - internal trend is bullish OR price sits inside a live bullish OB
        Short signal mirrors this for -1.
        Returns direction=0 if no fresh signal on the latest closed bar.
        """
        df_ce = self.chandelier_exit(df)
        ce_dir_now = int(df_ce["ce_dir"].iat[-1])
        ce_dir_prev = int(df_ce["ce_dir"].iat[-2]) if len(df_ce) > 1 else ce_dir_now

        internal_trend = self.detect_internal_trend(df)
        self.update_order_blocks(df)

        price = float(df["close"].iat[-1])
        ts = df.index[-1]

        ce_flip_long = ce_dir_prev == -1 and ce_dir_now == 1
        ce_flip_short = ce_dir_prev == 1 and ce_dir_now == -1

        if ce_flip_long and (internal_trend == 1 or self.price_in_bullish_ob(price)):
            self.last_ce_dir = ce_dir_now
            return Signal(ts, 1, "CE flipped bullish + bullish structure/OB confluence", price)

        if ce_flip_short and (internal_trend == -1 or self.price_in_bearish_ob(price)):
            self.last_ce_dir = ce_dir_now
            return Signal(ts, -1, "CE flipped bearish + bearish structure/OB confluence", price)

        self.last_ce_dir = ce_dir_now
        return Signal(ts, 0, "no signal", price)

    def current_ce_dir(self, df: pd.DataFrame) -> int:
        return int(self.chandelier_exit(df)["ce_dir"].iat[-1])

# ==============================================================================
# DHAN BROKER WRAPPER
# ==============================================================================

class DhanBroker:
    """
    Thin wrapper around the `dhanhq` package. All real-order paths are gated
    by cfg.DRY_RUN - in dry run, orders are simulated and logged only.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dhan = None
        self._security_cache: Optional[pd.DataFrame] = None
        if not cfg.DRY_RUN:
            self._connect()
        else:
            log.info("DRY_RUN=True -> paper trading mode. No live connection required for orders, "
                      "but live market data still needs valid credentials.")
            self._connect(quiet_on_fail=True)

    def _connect(self, quiet_on_fail: bool = False):
        try:
            from dhanhq import DhanContext, dhanhq
        except ImportError:
            msg = "dhanhq package not installed. Run: pip install dhanhq"
            if quiet_on_fail:
                log.warning(msg)
                return
            raise RuntimeError(msg)

        if not self.cfg.DHAN_CLIENT_ID or not self.cfg.DHAN_ACCESS_TOKEN:
            msg = "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set (env vars or Config)."
            if quiet_on_fail:
                log.warning(msg)
                return
            raise RuntimeError(msg)

        ctx = DhanContext(self.cfg.DHAN_CLIENT_ID, self.cfg.DHAN_ACCESS_TOKEN)
        self.dhan = dhanhq(ctx)
        log.info("Connected to DhanHQ.")

    # ---------------- Market data -----------------------------------------
    def get_intraday_candles(self, security_id: str, exchange_segment: str,
                              instrument_type: str = "INDEX",
                              days_back: int = 5) -> pd.DataFrame:
        """Fetch recent intraday minute data and resample to the configured
        timeframe. Falls back to a clear error if the API call fails."""
        if self.dhan is None:
            raise RuntimeError("Not connected to Dhan - cannot fetch live data.")

        to_date = dt.datetime.now()
        from_date = to_date - dt.timedelta(days=days_back)
        resp = self.dhan.intraday_minute_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        df = pd.DataFrame(data)
        if df.empty:
            return df
        rename_map = {"start_Time": "timestamp", "timestamp": "timestamp"}
        if "timestamp" not in df.columns and "start_Time" in df.columns:
            df = df.rename(columns={"start_Time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                                 "close": "close", "volume": "volume"})
        ohlc_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in ohlc_cols if c in df.columns]].astype(float)

        tf = f"{self.cfg.TIMEFRAME_MINUTES}min"
        resampled = df.resample(tf).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()
        return resampled

    def get_option_chain(self, under_security_id: str, under_exchange_segment: str,
                          expiry: str) -> dict:
        if self.dhan is None:
            raise RuntimeError("Not connected to Dhan.")
        return self.dhan.option_chain(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            expiry=expiry,
        )

    def get_nearest_expiry(self, under_security_id: str, under_exchange_segment: str) -> str:
        if self.dhan is None:
            raise RuntimeError("Not connected to Dhan.")
        resp = self.dhan.expiry_list(under_security_id=under_security_id,
                                      under_exchange_segment=under_exchange_segment)
        dates = resp.get("data", resp) if isinstance(resp, dict) else resp
        if isinstance(dates, list) and dates:
            return sorted(dates)[0]
        raise RuntimeError(f"Could not retrieve expiry list: {resp}")

    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        if self.dhan is None:
            raise RuntimeError("Not connected to Dhan.")
        resp = self.dhan.ohlc_data(securities={exchange_segment: [int(security_id)]})
        try:
            data = resp["data"][exchange_segment][str(security_id)]
            return float(data["last_price"])
        except Exception:
            raise RuntimeError(f"Unexpected LTP response shape: {resp}")

    # ---------------- Orders ------------------------------------------------
    def place_option_order(self, security_id: str, transaction_type: str,
                            quantity: int, order_type: str = "MARKET",
                            price: float = 0.0) -> dict:
        """
        transaction_type: "BUY" or "SELL"
        Gated by DRY_RUN: returns a simulated fill instead of hitting the API.
        """
        if self.cfg.DRY_RUN:
            sim_id = f"SIM-{int(time.time()*1000)}"
            log.info(f"[DRY RUN] Would place {transaction_type} order: "
                     f"security_id={security_id} qty={quantity} type={order_type} price={price}")
            return {"orderId": sim_id, "orderStatus": "SIMULATED"}

        if self.dhan is None:
            raise RuntimeError("Not connected to Dhan - cannot place a live order.")

        resp = self.dhan.place_order(
            security_id=security_id,
            exchange_segment=self.cfg.OPTION_EXCHANGE_SEGMENT,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type="INTRA",
            price=price,
        )
        log.info(f"LIVE ORDER PLACED: {resp}")
        return resp

    def get_positions(self) -> list:
        if self.cfg.DRY_RUN or self.dhan is None:
            return []
        return self.dhan.get_positions()


# ==============================================================================
# OPTION SELECTION HELPERS
# ==============================================================================

def round_to_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def find_option_contract(option_chain_resp: dict, target_strike: int,
                          option_type: str) -> Optional[dict]:
    """
    Parses the Dhan option_chain() response and returns the contract dict
    (with security_id, last_price, greeks, etc.) matching target_strike and
    option_type ("CE"/"PE"). Response shape per DhanHQ v2 option_chain docs:
        { "data": { "oc": { "<strike>": {"ce": {...}, "pe": {...}}, ... },
                    "last_price": <spot> }, "status": "success" }
    """
    try:
        oc = option_chain_resp["data"]["oc"]
    except Exception:
        log.error(f"Unexpected option chain response shape: {option_chain_resp}")
        return None

    key_candidates = [str(target_strike), f"{target_strike:.6f}", f"{target_strike}.000000"]
    strike_data = None
    for k in key_candidates:
        if k in oc:
            strike_data = oc[k]
            break
    if strike_data is None:
        # fall back to nearest available strike key
        try:
            numeric_keys = {float(k): k for k in oc.keys()}
            nearest = min(numeric_keys.keys(), key=lambda x: abs(x - target_strike))
            strike_data = oc[numeric_keys[nearest]]
        except Exception:
            return None

    leg = strike_data.get("ce" if option_type == "CE" else "pe")
    return leg

# ==============================================================================
# RISK MANAGER  -- this is what actually bounds drawdown; the signal logic
# alone cannot guarantee any win rate or drawdown ceiling.
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
            log.info("New trading day - risk counters reset.")

    def record_trade_result(self, pnl: float):
        self.realized_pnl_today += pnl
        self.trades_today += 1
        max_loss = -(self.cfg.MAX_DAILY_LOSS_PCT / 100.0) * self.day_start_capital
        if self.realized_pnl_today <= max_loss:
            self.trading_halted_today = True
            log.warning(f"DAILY MAX LOSS HIT ({self.realized_pnl_today:.2f}). "
                        f"Trading halted for the rest of the day.")

    def can_trade(self) -> Tuple[bool, str]:
        self.new_day_if_needed()
        if self.trading_halted_today:
            return False, "daily loss limit hit"
        if self.trades_today >= self.cfg.MAX_TRADES_PER_DAY:
            return False, "max trades/day reached"
        now = dt.datetime.now().time()
        if now >= self.cfg.SQUARE_OFF_TIME:
            return False, "past square-off time"
        return True, "ok"

    def position_size(self, entry_premium: float, stop_loss_premium: float,
                       lot_size: int) -> int:
        """
        Size by: risk_amount = CAPITAL * RISK_PER_TRADE_PCT
                 risk_per_lot = (entry_premium - stop_loss_premium) * lot_size
                 lots = floor(risk_amount / risk_per_lot), capped by MAX_LOTS_PER_TRADE
        """
        risk_amount = (self.cfg.RISK_PER_TRADE_PCT / 100.0) * self.cfg.CAPITAL
        risk_per_lot = max(entry_premium - stop_loss_premium, 0.01) * lot_size
        lots = max(int(risk_amount // risk_per_lot), 1)
        lots = min(lots, self.cfg.MAX_LOTS_PER_TRADE)
        return lots * lot_size


# ==============================================================================
# TRADE LOGGER (CSV) + simple JSON state persistence
# ==============================================================================

class TradeLogger:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["entry_time", "exit_time", "direction", "symbol",
                                  "strike", "option_type", "entry_premium",
                                  "exit_premium", "quantity", "pnl", "exit_reason"])

    def log(self, pos: OpenPosition, exit_time, exit_premium: float,
             pnl: float, exit_reason: str):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pos.entry_time, exit_time,
                              "LONG" if pos.direction == 1 else "SHORT",
                              pos.option_symbol, pos.strike, pos.option_type,
                              pos.entry_premium, exit_premium, pos.quantity,
                              pnl, exit_reason])


# ==============================================================================
# LIVE WEB DASHBOARD
#
# A small Flask app served in a background thread of the same process.
# The trading engine pushes snapshots into DashboardState (thread-safe via a
# lock); the browser polls GET /api/state every few seconds and redraws.
# No external JS chart library is used - the chart is plain <canvas>, so the
# dashboard has zero internet dependency beyond loading the page itself.
# ==============================================================================

class DashboardState:
    """Thread-safe snapshot of everything the dashboard needs to render."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "mode": "PAPER",
            "underlying": "",
            "last_update": None,
            "candles": [],          # [{t, o, h, l, c}]
            "bull_obs": [],         # [{top, bottom}]
            "bear_obs": [],
            "signals": [],          # [{t, price, dir, reason}]
            "position": None,       # current open position or None
            "risk": {
                "capital": 0, "pnl_today": 0, "trades_today": 0,
                "max_trades": 0, "halted": False, "max_daily_loss_pct": 0,
            },
            "trades": [],           # closed trades, most recent first
            "equity_curve": [],     # [{t, equity}]
            "status_message": "starting...",
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

  // gridlines + price labels
  ctx.strokeStyle = '#1d2429'; ctx.fillStyle = '#6f7c85'; ctx.font = '10px monospace';
  for (let i = 0; i <= 4; i++) {
    const p = minP + (range * i / 4);
    const y = yOf(p);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(p.toFixed(0), 4, y + 3);
  }

  // order block zones
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

  // candles
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

  // signal markers
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
        <div class="position-row"><span class="k">Stop ₹</span><span>${fmt(p.stop_loss_premium)}</span></div>
        <div class="position-row"><span class="k">Target ₹</span><span>${p.target_premium === Infinity ? '—' : fmt(p.target_premium)}</span></div>
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


def create_dashboard_app(state: DashboardState):
    from flask import Flask, jsonify, Response

    app = Flask(__name__)
    # Quiet down Flask's own request logging so it doesn't spam our log file.
    flask_logger = logging.getLogger("werkzeug")
    flask_logger.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    return app


def start_dashboard_thread(cfg: Config, state: DashboardState):
    try:
        app = create_dashboard_app(state)
    except ImportError:
        log.warning("Flask not installed - dashboard disabled. Run: pip install flask")
        return None

    def _run():
        app.run(host=cfg.DASHBOARD_HOST, port=cfg.DASHBOARD_PORT,
                 debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    log.info(f"Dashboard running at http://{cfg.DASHBOARD_HOST}:{cfg.DASHBOARD_PORT}")
    return t


# ==============================================================================
# LIVE / PAPER TRADING ENGINE
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
        if self.dashboard_state is None:
            return
        cfg = self.cfg

        candles, bull_obs, bear_obs = [], [], []
        if df is not None and not df.empty:
            tail = df.tail(cfg.DASHBOARD_CHART_BARS)
            candles = [
                {"t": str(idx), "o": float(r.open), "h": float(r.high),
                 "l": float(r.low), "c": float(r.close)}
                for idx, r in tail.iterrows()
            ]
            bull_obs = [{"top": ob.top, "bottom": ob.bottom} for ob in self.engine.bull_obs]
            bear_obs = [{"top": ob.top, "bottom": ob.bottom} for ob in self.engine.bear_obs]

        position = None
        if self.open_position:
            p = self.open_position
            position = {
                "direction": p.direction, "strike": p.strike, "option_type": p.option_type,
                "quantity": p.quantity, "entry_premium": p.entry_premium,
                "stop_loss_premium": p.stop_loss_premium,
                "target_premium": p.target_premium if p.target_premium != float("inf") else None,
                "entry_time": str(p.entry_time),
            }

        max_loss_pct = self.cfg.MAX_DAILY_LOSS_PCT
        self.dashboard_state.update(
            mode="PAPER" if cfg.DRY_RUN else "LIVE",
            underlying=cfg.UNDERLYING,
            candles=candles,
            bull_obs=bull_obs,
            bear_obs=bear_obs,
            signals=self.recent_signals[-20:],
            position=position,
            risk={
                "capital": cfg.CAPITAL,
                "pnl_today": self.risk.realized_pnl_today,
                "trades_today": self.risk.trades_today,
                "max_trades": cfg.MAX_TRADES_PER_DAY,
                "halted": self.risk.trading_halted_today,
                "max_daily_loss_pct": max_loss_pct,
            },
            trades=self._trade_history[-50:][::-1] if hasattr(self, "_trade_history") else [],
            equity_curve=self.equity_curve[-200:],
            status_message=status,
        )

    def _select_and_enter(self, signal: Signal):
        cfg = self.cfg
        can, reason = self.risk.can_trade()
        if not can:
            log.info(f"Skipping signal - {reason}")
            return

        sec_id, seg = self._underlying_ids()
        try:
            expiry = self.broker.get_nearest_expiry(sec_id, seg)
            chain = self.broker.get_option_chain(sec_id, seg, expiry)
        except Exception as e:
            log.error(f"Could not fetch option chain: {e}")
            return

        spot = signal.price
        step = cfg.STRIKE_STEP[cfg.UNDERLYING]
        atm_strike = round_to_strike(spot, step)
        option_type = "CE" if signal.direction == 1 else "PE"

        leg = find_option_contract(chain, atm_strike, option_type)
        if leg is None:
            log.error(f"No contract found for strike {atm_strike} {option_type}")
            return

        try:
            entry_premium = float(leg["last_price"])
            option_security_id = str(leg["security_id"])
        except Exception:
            log.error(f"Malformed option leg data: {leg}")
            return

        lot_size = cfg.LOT_SIZE[cfg.UNDERLYING]
        stop_loss_premium = entry_premium * (1 - cfg.STOP_LOSS_PCT_OF_PREMIUM / 100.0)
        target_premium = (entry_premium * (1 + cfg.TARGET_PCT_OF_PREMIUM / 100.0)
                           if cfg.TARGET_PCT_OF_PREMIUM > 0 else float("inf"))

        quantity = self.risk.position_size(entry_premium, stop_loss_premium, lot_size)

        resp = self.broker.place_option_order(option_security_id, "BUY", quantity)

        self.open_position = OpenPosition(
            direction=signal.direction,
            option_security_id=option_security_id,
            option_symbol=leg.get("tradingsymbol", f"{cfg.UNDERLYING}{atm_strike}{option_type}"),
            strike=atm_strike,
            option_type=option_type,
            entry_premium=entry_premium,
            quantity=quantity,
            stop_loss_premium=stop_loss_premium,
            target_premium=target_premium,
            entry_time=signal.timestamp,
            order_id=resp.get("orderId"),
        )
        log.info(f"ENTERED {option_type} {atm_strike} qty={quantity} @ {entry_premium} "
                 f"(SL={stop_loss_premium:.2f}, TGT={target_premium:.2f}) reason='{signal.reason}'")
        self._push_dashboard(status="position entered")

    def _check_exit(self, df_underlying: pd.DataFrame):
        if self.open_position is None:
            return
        pos = self.open_position

        try:
            current_premium = self.broker.get_ltp(pos.option_security_id, self.cfg.OPTION_EXCHANGE_SEGMENT)
        except Exception as e:
            log.error(f"Could not fetch option LTP for exit check: {e}")
            return

        exit_reason = None
        if current_premium <= pos.stop_loss_premium:
            exit_reason = "stop_loss"
        elif current_premium >= pos.target_premium:
            exit_reason = "target"
        elif self.cfg.TRAIL_WITH_CHANDELIER:
            ce_dir = self.engine.current_ce_dir(df_underlying)
            if (pos.direction == 1 and ce_dir == -1) or (pos.direction == -1 and ce_dir == 1):
                exit_reason = "chandelier_flip"

        now = dt.datetime.now().time()
        if now >= self.cfg.SQUARE_OFF_TIME:
            exit_reason = exit_reason or "square_off"

        if exit_reason:
            self._exit_position(pos, current_premium, exit_reason)

    def _exit_position(self, pos: OpenPosition, exit_premium: float, reason: str):
        self.broker.place_option_order(pos.option_security_id, "SELL", pos.quantity)
        pnl = (exit_premium - pos.entry_premium) * pos.quantity
        self.risk.record_trade_result(pnl)
        self.trade_log.log(pos, dt.datetime.now(), exit_premium, pnl, reason)
        log.info(f"EXITED {pos.option_type} {pos.strike} @ {exit_premium} "
                 f"reason={reason} pnl={pnl:.2f}")

        self.equity += pnl
        now_iso = dt.datetime.now().isoformat()
        self._trade_history.append({
            "entry_time": str(pos.entry_time), "exit_time": now_iso,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "strike": pos.strike, "option_type": pos.option_type,
            "quantity": pos.quantity, "entry_premium": pos.entry_premium,
            "exit_premium": exit_premium, "pnl": pnl, "exit_reason": reason,
        })
        self.equity_curve.append({"t": now_iso, "equity": self.equity})
        self.open_position = None

    def run_forever(self):
        cfg = self.cfg
        log.info(f"Starting engine | DRY_RUN={cfg.DRY_RUN} | "
                 f"underlying={cfg.UNDERLYING} | timeframe={cfg.TIMEFRAME_MINUTES}min")
        sec_id, seg = self._underlying_ids()

        if self.dashboard_state is not None:
            start_dashboard_thread(cfg, self.dashboard_state)

        while True:
            try:
                now_t = dt.datetime.now().time()
                if now_t >= cfg.SQUARE_OFF_TIME and self.open_position is not None:
                    df = self.broker.get_intraday_candles(sec_id, seg, "INDEX")
                    self._check_exit(df)

                df = self.broker.get_intraday_candles(sec_id, seg, "INDEX")
                if df.empty or len(df) < max(cfg.SWING_LENGTH, cfg.CE_ATR_PERIOD) + 5:
                    log.info("Not enough candle data yet, waiting...")
                    self._push_dashboard(df, status="waiting for enough candle history")
                    time.sleep(cfg.POLL_SECONDS)
                    continue

                latest_bar_time = df.index[-1]
                if self.open_position is not None:
                    self._check_exit(df)

                if latest_bar_time != self.last_seen_bar:
                    self.last_seen_bar = latest_bar_time
                    signal = self.engine.generate_signal(df)
                    if signal.direction != 0:
                        self.recent_signals.append({
                            "t": str(signal.timestamp), "price": signal.price,
                            "dir": signal.direction, "reason": signal.reason,
                        })
                        if self.open_position is None:
                            self._select_and_enter(signal)

                self._push_dashboard(df, status="running")
                time.sleep(cfg.POLL_SECONDS)

            except KeyboardInterrupt:
                log.info("Stopped by user.")
                break
            except Exception as e:
                log.error(f"Engine loop error: {e}\n{traceback.format_exc()}")
                time.sleep(cfg.POLL_SECONDS)

# ==============================================================================
# BACKTESTER
#
# Approximates option P&L by applying the underlying's % move to a synthetic
# premium using a simple delta proxy (since historical option-chain premium
# history is not reliably available via the API for old dates). This is a
# SIMPLIFICATION - use it to sanity-check signal quality and rough risk
# behaviour, not as a precise P&L simulator. For real validation, backtest
# against actual historical option premiums if you can source them.
# ==============================================================================

@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_spot: float
    exit_spot: float
    pnl_points: float
    pnl_pct_of_premium: float
    exit_reason: str


def backtest(df: pd.DataFrame, cfg: Config) -> Tuple[List[BacktestTrade], pd.DataFrame]:
    engine = StructureEngine(cfg)
    trades: List[BacktestTrade] = []
    open_trade: Optional[dict] = None

    min_bars = max(cfg.SWING_LENGTH, cfg.CE_ATR_PERIOD) + 5
    equity_curve = []
    equity = cfg.CAPITAL
    assumed_delta = 0.5  # rough ATM option delta proxy for points->premium-% conversion

    for i in range(min_bars, len(df)):
        window = df.iloc[:i + 1]
        ts = window.index[-1]
        price = float(window["close"].iat[-1])

        if open_trade is not None:
            ce_dir = engine.current_ce_dir(window)
            move_pts = price - open_trade["entry_spot"] if open_trade["direction"] == 1 \
                else open_trade["entry_spot"] - price
            premium_change_pct = (move_pts * assumed_delta / open_trade["entry_spot"]) * 100 \
                * (open_trade["entry_spot"] / 100)  # scaled heuristic, see note above
            # Simplify: treat % underlying move * delta as % premium move
            underlying_move_pct = (move_pts / open_trade["entry_spot"]) * 100
            premium_move_pct = underlying_move_pct * assumed_delta * 5  # leverage proxy for OTM/ATM options

            exit_reason = None
            if premium_move_pct <= -cfg.STOP_LOSS_PCT_OF_PREMIUM:
                exit_reason = "stop_loss"
            elif cfg.TARGET_PCT_OF_PREMIUM > 0 and premium_move_pct >= cfg.TARGET_PCT_OF_PREMIUM:
                exit_reason = "target"
            elif cfg.TRAIL_WITH_CHANDELIER and (
                (open_trade["direction"] == 1 and ce_dir == -1) or
                (open_trade["direction"] == -1 and ce_dir == 1)
            ):
                exit_reason = "chandelier_flip"

            if exit_reason:
                pnl_pct = max(premium_move_pct, -cfg.STOP_LOSS_PCT_OF_PREMIUM)
                trades.append(BacktestTrade(
                    entry_time=open_trade["entry_time"], exit_time=ts,
                    direction=open_trade["direction"], entry_spot=open_trade["entry_spot"],
                    exit_spot=price, pnl_points=move_pts,
                    pnl_pct_of_premium=pnl_pct, exit_reason=exit_reason,
                ))
                trade_risk_amount = (cfg.RISK_PER_TRADE_PCT / 100.0) * equity
                pnl_amount = trade_risk_amount * (pnl_pct / cfg.STOP_LOSS_PCT_OF_PREMIUM)
                equity += pnl_amount
                open_trade = None

        if open_trade is None:
            signal = engine.generate_signal(window)
            if signal.direction != 0:
                open_trade = {"entry_time": ts, "entry_spot": price, "direction": signal.direction}

        equity_curve.append({"time": ts, "equity": equity})

    equity_df = pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame()
    return trades, equity_df


def summarize_backtest(trades: List[BacktestTrade], equity_df: pd.DataFrame, cfg: Config):
    if not trades:
        print("No trades generated over this period.")
        return

    wins = [t for t in trades if t.pnl_pct_of_premium > 0]
    losses = [t for t in trades if t.pnl_pct_of_premium <= 0]
    win_rate = 100 * len(wins) / len(trades)

    if not equity_df.empty:
        running_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - running_max) / running_max * 100
        max_dd = drawdown.min()
    else:
        max_dd = float("nan")

    total_return_pct = ((equity_df["equity"].iat[-1] / cfg.CAPITAL) - 1) * 100 if not equity_df.empty else float("nan")

    print("=" * 60)
    print(" BACKTEST SUMMARY (approximate option P&L - see caveats above)")
    print("=" * 60)
    print(f" Total trades        : {len(trades)}")
    print(f" Win rate            : {win_rate:.1f}%")
    print(f" Max drawdown        : {max_dd:.2f}%")
    print(f" Total return        : {total_return_pct:.2f}%")
    print(f" Avg win (%premium)  : {np.mean([t.pnl_pct_of_premium for t in wins]):.1f}%" if wins else " Avg win: n/a")
    print(f" Avg loss (%premium) : {np.mean([t.pnl_pct_of_premium for t in losses]):.1f}%" if losses else " Avg loss: n/a")
    print("=" * 60)
    print(" NOTE: this win-rate/drawdown estimate uses a simplified delta-proxy")
    print(" P&L model, NOT real historical option premiums. Treat as directional")
    print(" signal-quality feedback, not a guarantee of live performance.")
    print("=" * 60)


def run_backtest_cli(cfg: Config, from_date: str, to_date: str):
    broker = DhanBroker(cfg)
    sec_id, seg = cfg.UNDERLYING_SECURITY_ID[cfg.UNDERLYING], cfg.UNDERLYING_EXCHANGE_SEGMENT
    log.info(f"Fetching historical data for {cfg.UNDERLYING} {from_date} -> {to_date}")
    try:
        resp = broker.dhan.historical_daily_data(
            security_id=sec_id, exchange_segment=seg, instrument_type="INDEX",
            from_date=from_date, to_date=to_date,
        )
    except Exception as e:
        log.error(f"Historical data fetch failed: {e}")
        return
    data = resp.get("data", resp) if isinstance(resp, dict) else resp
    df = pd.DataFrame(data)
    if df.empty:
        log.error("No historical data returned for the given range.")
        return
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    trades, equity_df = backtest(df, cfg)
    summarize_backtest(trades, equity_df, cfg)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SMC+OB+Chandelier Exit options bot for DhanHQ")
    p.add_argument("--live", action="store_true", help="Place REAL orders (overrides DRY_RUN=True default)")
    p.add_argument("--backtest", action="store_true", help="Run backtest instead of live/paper loop")
    p.add_argument("--underlying", default=CFG.UNDERLYING, choices=["NIFTY", "BANKNIFTY"])
    p.add_argument("--from", dest="from_date", default=None, help="Backtest start date YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", default=None, help="Backtest end date YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=CFG.CAPITAL)
    p.add_argument("--risk-pct", type=float, default=CFG.RISK_PER_TRADE_PCT)
    p.add_argument("--no-dashboard", action="store_true", help="Disable the live web dashboard")
    p.add_argument("--port", type=int, default=CFG.DASHBOARD_PORT, help="Dashboard port (default 8765)")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.UNDERLYING = args.underlying
    CFG.CAPITAL = args.capital
    CFG.RISK_PER_TRADE_PCT = args.risk_pct
    CFG.DASHBOARD_ENABLED = not args.no_dashboard
    CFG.DASHBOARD_PORT = args.port

    if args.live:
        CFG.DRY_RUN = False
        log.warning("=" * 60)
        log.warning(" LIVE MODE ENABLED - REAL ORDERS WILL BE PLACED.")
        log.warning(" Ctrl+C within 5 seconds to abort.")
        log.warning("=" * 60)
        time.sleep(5)

    if args.backtest:
        if not args.from_date or not args.to_date:
            print("--backtest requires --from YYYY-MM-DD and --to YYYY-MM-DD")
            sys.exit(1)
        run_backtest_cli(CFG, args.from_date, args.to_date)
        return

    engine = TradingEngine(CFG)
    engine.run_forever()


if __name__ == "__main__":
    main()