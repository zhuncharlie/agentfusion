"""R002: verify backtest.py's metrics against an independent hand calculation."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentfusion import Action, BaseAgent, OptimizerRegistry, Signal
from agentfusion.backtest import run_backtest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class BuyHoldAgent(BaseAgent):
    def decide(self, obs: dict) -> Signal:
        return Signal(Action.BUY if len(obs["history"]) == 1 else Action.HOLD)


def _price_df(closes):
    dates = pd.date_range("2023-01-02", periods=len(closes), freq="B")
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000}
    )


def test_buy_hold_matches_hand_calculation():
    df = _price_df([100, 102, 101, 105, 103])
    metrics = run_backtest(BuyHoldAgent(), df, ticker="SYNTH")

    assert metrics["return"] == pytest.approx(0.02897, abs=1e-5)
    assert metrics["mdd"] == pytest.approx(-0.0190476, abs=1e-6)
    assert metrics["sharpe"] == pytest.approx(4.411184, abs=1e-4)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["calmar"] == pytest.approx(1.520925, abs=1e-4)


def test_buy_hold_spy_benchmark_via_registry():
    OptimizerRegistry.register("buy_hold_test")(BuyHoldAgent)
    df = pd.read_csv(DATA_DIR / "SPY.csv", parse_dates=["date"])

    agent = OptimizerRegistry.get("buy_hold_test")()
    metrics = run_backtest(agent, df, ticker="SPY")

    # Independent hand calculation, not reusing backtest.py's code path.
    cash = 100_000.0
    shares = (cash - cash * 0.001) / df["close"].iloc[0]
    equity = shares * df["close"]
    expected_return = equity.iloc[-1] / cash - 1
    running_max = equity.cummax()
    expected_mdd = (equity / running_max - 1).min()

    assert metrics["return"] == pytest.approx(expected_return, abs=1e-9)
    assert metrics["mdd"] == pytest.approx(expected_mdd, abs=1e-9)
    assert math.isfinite(metrics["sharpe"])
    assert metrics["mdd"] <= 0
