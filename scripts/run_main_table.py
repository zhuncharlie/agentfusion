"""R007: full main-results table -- BuyHold vs FinRL vs TradingAgents vs MajorityVoteEnsemble."""
import pandas as pd

import agentfusion.agents.buy_hold  # noqa: F401
import agentfusion.agents.ensemble  # noqa: F401
import agentfusion.agents.finrl_ppo  # noqa: F401
import agentfusion.agents.trading_agents  # noqa: F401
from agentfusion.backtest import run_backtest
from agentfusion.registry import OptimizerRegistry

TICKERS = ["AAPL", "MSFT", "NVDA"]
TEST_START = "2023-01-01"
SYSTEMS = ["buy_hold", "finrl_ppo", "trading_agents", "majority_vote_ensemble"]

if __name__ == "__main__":
    rows = []
    for ticker in TICKERS:
        df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["date"])
        test_df = df[df["date"] >= TEST_START].reset_index(drop=True)

        for system_name in SYSTEMS:
            agent_cls = OptimizerRegistry.get(system_name)
            agent = agent_cls() if system_name == "buy_hold" else agent_cls(ticker=ticker)
            result = run_backtest(agent, test_df, ticker=ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "system": system_name,
                    "sharpe": result["sharpe"],
                    "return": result["return"],
                    "mdd": result["mdd"],
                    "calmar": result["calmar"],
                    "win_rate": result["win_rate"],
                }
            )
            print(
                f"{ticker:6s} {system_name:24s} Sharpe={result['sharpe']:.3f} "
                f"Return={result['return']:.3%} MDD={result['mdd']:.3%}"
            )

    table = pd.DataFrame(rows)
    table.to_csv("results/main_table.csv", index=False)
    print("\nWritten to results/main_table.csv")
    print(table.to_string(index=False))
