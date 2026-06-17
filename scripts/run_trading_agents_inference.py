"""R005/R006: generate (and cache) TradingAgentsAgent signals for the full test period."""
import json

import pandas as pd

import agentfusion.agents.trading_agents as ta_module
from agentfusion.backtest import run_backtest
from agentfusion.registry import OptimizerRegistry

TICKERS = ["AAPL", "MSFT", "NVDA"]
TEST_START = "2023-01-01"

if __name__ == "__main__":
    metrics_table = {}

    for ticker in TICKERS:
        df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["date"])
        test_df = df[df["date"] >= TEST_START].reset_index(drop=True)

        agent_cls = OptimizerRegistry.get("trading_agents")
        agent = agent_cls(ticker=ticker)
        result = run_backtest(agent, test_df, ticker=ticker)

        metrics_table[ticker] = {k: v for k, v in result.items() if k not in ("equity_curve", "signals")}
        action_counts = result["signals"]["action"].value_counts().to_dict()
        print(f"{ticker}: {action_counts}  Sharpe={result['sharpe']:.3f} Return={result['return']:.3%} MDD={result['mdd']:.3%}")

    with open("results/trading_agents_metrics.json", "w") as f:
        json.dump(metrics_table, f, indent=2, default=float)

    cost_log = ta_module._load_cost_log()
    print(f"\nTotal API cost: ${cost_log['total_usd']:.4f} over {cost_log['calls']} calls")
    print("Signals cached at signals/trading_agents.csv")
    print("Metrics written to results/trading_agents_metrics.json")
