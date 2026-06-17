"""Minimal end-to-end demo: a third-party agent registers and runs in <=10 lines."""
import pandas as pd

from agentfusion import Action, BaseAgent, OptimizerRegistry, Signal
from agentfusion.backtest import run_backtest

# --- registration: this is the part a third-party contributor writes (<=10 lines) ---
@OptimizerRegistry.register("buy_hold")
class BuyHoldAgent(BaseAgent):
    def decide(self, obs: dict) -> Signal:
        is_first_bar = len(obs["history"]) == 1
        return Signal(Action.BUY if is_first_bar else Action.HOLD)
# --- end registration ---


def _make_demo_prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    close = 100 + pd.Series(range(60), dtype=float) * 0.3
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )


if __name__ == "__main__":
    price_df = _make_demo_prices()
    agent = OptimizerRegistry.get("buy_hold")()
    metrics = run_backtest(agent, price_df, ticker="DEMO")
    print(f"Sharpe={metrics['sharpe']:.3f}  Return={metrics['return']:.3%}  MDD={metrics['mdd']:.3%}")
