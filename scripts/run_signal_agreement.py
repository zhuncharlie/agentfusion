"""R008 (Block 3): RL vs LLM signal-agreement analysis. Exploratory, not confirmatory."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import agentfusion.agents.buy_hold  # noqa: F401
import agentfusion.agents.ensemble  # noqa: F401
import agentfusion.agents.finrl_ppo  # noqa: F401
import agentfusion.agents.trading_agents  # noqa: F401
from agentfusion.backtest import run_backtest
from agentfusion.registry import OptimizerRegistry

TICKERS = ["AAPL", "MSFT", "NVDA"]
TEST_START = "2023-01-01"

if __name__ == "__main__":
    rl_signals = pd.read_csv("signals/finrl_ppo.csv", parse_dates=["date"])
    llm_signals = pd.read_csv("signals/trading_agents.csv", parse_dates=["date"])

    summary = {}
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(10, 4 * len(TICKERS)), sharex=False)

    for i, ticker in enumerate(TICKERS):
        price_df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["date"])
        test_df = price_df[price_df["date"] >= TEST_START].reset_index(drop=True)
        daily_return = test_df["close"].pct_change().fillna(0.0)

        rl = rl_signals[rl_signals["ticker"] == ticker].set_index("date")["action"]
        llm = llm_signals[llm_signals["ticker"] == ticker].set_index("date")["action"]

        merged = pd.DataFrame({"date": test_df["date"], "ret": daily_return})
        merged["rl_action"] = merged["date"].map(rl)
        merged["llm_action"] = merged["date"].map(llm)
        merged["agree"] = merged["rl_action"] == merged["llm_action"]

        agree_ret = merged.loc[merged["agree"], "ret"]
        disagree_ret = merged.loc[~merged["agree"], "ret"]
        t_stat, p_value = stats.ttest_ind(agree_ret, disagree_ret, equal_var=False, nan_policy="omit")

        summary[ticker] = {
            "agreement_rate": float(merged["agree"].mean()),
            "n_agree_days": int(merged["agree"].sum()),
            "n_disagree_days": int((~merged["agree"]).sum()),
            "mean_return_agree": float(agree_ret.mean()),
            "mean_return_disagree": float(disagree_ret.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "note": "exploratory, not confirmatory (small n, no multiple-comparison or clustered-SE correction)",
        }
        print(
            f"{ticker}: agreement_rate={summary[ticker]['agreement_rate']:.1%}  "
            f"mean_ret_agree={summary[ticker]['mean_return_agree']:.4%}  "
            f"mean_ret_disagree={summary[ticker]['mean_return_disagree']:.4%}  "
            f"t={t_stat:.3f} p={p_value:.3f}"
        )

        ensemble_cls = OptimizerRegistry.get("majority_vote_ensemble")
        ensemble_result = run_backtest(ensemble_cls(ticker=ticker), test_df, ticker=ticker)
        equity = ensemble_result["equity_curve"]

        ax = axes[i]
        ax.plot(equity.index, equity.values, label="Ensemble equity", color="black", linewidth=1.2)
        disagree_dates = merged.loc[~merged["agree"], "date"]
        for d in disagree_dates:
            ax.axvspan(d, d + pd.Timedelta(days=1), color="grey", alpha=0.25, linewidth=0)
        ax.set_title(f"{ticker}: ensemble equity, grey = RL/LLM signal disagreement")
        ax.legend(loc="upper left")

    fig.tight_layout()
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("figures/signal_agreement.png", dpi=120)

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/signal_agreement.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFigure written to figures/signal_agreement.png")
    print("Summary written to results/signal_agreement.json")
