"""R009 pluggability self-check: simulates a third-party contributor's first agent.

Written using only BaseAgent's documented obs contract (ticker/date/row/history) --
no knowledge of agentfusion internals beyond what examples/dummy_agent.py shows.
"""
import pandas as pd

from agentfusion import Action, BaseAgent, OptimizerRegistry, Signal
from agentfusion.backtest import run_backtest

# --- registration: what a third-party contributor writes (<=10 lines) ---
@OptimizerRegistry.register("sma_crossover")
class ExampleThirdPartyAgent(BaseAgent):
    def decide(self, obs: dict) -> Signal:
        closes = obs["history"]["close"]
        if len(closes) < 10:
            return Signal(Action.HOLD)
        return Signal(Action.BUY if closes.tail(5).mean() > closes.tail(10).mean() else Action.SELL)
# --- end registration ---


if __name__ == "__main__":
    df = pd.read_csv("data/raw/AAPL.csv", parse_dates=["date"])
    one_month = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2023-01-31")].reset_index(drop=True)

    agent = OptimizerRegistry.get("sma_crossover")()
    metrics = run_backtest(agent, one_month, ticker="AAPL")
    print(f"Sharpe={metrics['sharpe']:.3f}  Return={metrics['return']:.3%}  MDD={metrics['mdd']:.3%}")
