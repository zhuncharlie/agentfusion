"""R004: run FinRLPPOAgent (trained checkpoints) on the test period via the Registry."""
import json

import pandas as pd

import agentfusion.agents.finrl_ppo  # noqa: F401  (registers "finrl_ppo")
from agentfusion.backtest import run_backtest
from agentfusion.registry import OptimizerRegistry

TICKERS = ["AAPL", "MSFT", "NVDA"]
TEST_START = "2023-01-01"

if __name__ == "__main__":
    all_signals = []
    metrics_table = {}

    for ticker in TICKERS:
        df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["date"])
        test_df = df[df["date"] >= TEST_START].reset_index(drop=True)

        agent_cls = OptimizerRegistry.get("finrl_ppo")
        agent = agent_cls(ticker=ticker)
        result = run_backtest(agent, test_df, ticker=ticker)

        all_signals.append(result["signals"])
        metrics_table[ticker] = {k: v for k, v in result.items() if k not in ("equity_curve", "signals")}

        action_counts = result["signals"]["action"].value_counts().to_dict()
        print(f"{ticker}: {action_counts}  Sharpe={result['sharpe']:.3f} Return={result['return']:.3%} MDD={result['mdd']:.3%}")

    pd.concat(all_signals, ignore_index=True).to_csv("signals/finrl_ppo.csv", index=False)
    with open("results/finrl_ppo_metrics.json", "w") as f:
        json.dump(metrics_table, f, indent=2, default=float)

    print("\nSignals written to signals/finrl_ppo.csv")
    print("Metrics written to results/finrl_ppo_metrics.json")
