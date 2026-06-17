"""Minimal single-asset daily backtest engine for AgentFusion agents."""
import numpy as np
import pandas as pd

from agentfusion.base import Action, BaseAgent

ANNUALIZATION_FACTOR = 252
RISK_FREE_ANNUAL = 0.045


def run_backtest(
    agent: BaseAgent,
    price_df: pd.DataFrame,
    ticker: str,
    initial_cash: float = 100_000.0,
    transaction_cost: float = 0.001,
) -> dict:
    """Run a daily long/flat backtest.

    price_df: ascending-by-date DataFrame with columns
      ['date', 'open', 'high', 'low', 'close', 'volume'].
    The agent sees only `history` up to and including the current bar (no lookahead)
    and reacts at that same bar's close.
    """
    cash = initial_cash
    shares = 0.0
    equity_curve = []
    signals = []

    for i in range(len(price_df)):
        row = price_df.iloc[i]
        obs = {
            "ticker": ticker,
            "date": row["date"],
            "row": row,
            "history": price_df.iloc[: i + 1],
        }
        signal = agent.decide(obs)
        signals.append({"date": row["date"], "ticker": ticker, "action": signal.action.value, "confidence": signal.confidence})
        price = row["close"]

        if signal.action == Action.BUY and cash > 0:
            cost = cash * transaction_cost
            shares += (cash - cost) / price
            cash = 0.0
        elif signal.action == Action.SELL and shares > 0:
            proceeds = shares * price
            cash += proceeds - proceeds * transaction_cost
            shares = 0.0
        # HOLD: no-op

        equity_curve.append(cash + shares * price)

    equity_curve = pd.Series(equity_curve, index=pd.Index(price_df["date"], name="date"))
    daily_returns = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / initial_cash - 1
    mdd = _max_drawdown(equity_curve)

    return {
        "sharpe": _annualized_sharpe(daily_returns),
        "return": total_return,
        "mdd": mdd,
        "calmar": (total_return / abs(mdd)) if mdd != 0 else float("nan"),
        "win_rate": (daily_returns > 0).mean() if len(daily_returns) else float("nan"),
        "equity_curve": equity_curve,
        "signals": pd.DataFrame(signals),
    }


def _annualized_sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return float("nan")
    rf_daily = RISK_FREE_ANNUAL / ANNUALIZATION_FACTOR
    excess = daily_returns - rf_daily
    return (excess.mean() / excess.std()) * np.sqrt(ANNUALIZATION_FACTOR)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1).min()
