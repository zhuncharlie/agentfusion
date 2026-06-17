"""R003: train FinRLPPOAgent (per-ticker PPO) on the 2020-01-01~2022-12-31 training period."""
import json
import time
from pathlib import Path

import pandas as pd

from agentfusion.agents.finrl_ppo import train_finrl_ppo

TICKERS = ["AAPL", "MSFT", "NVDA"]
TRAIN_END = "2022-12-31"
TOTAL_TIMESTEPS = 60_000
LOG_PATH = Path("checkpoints/finrl_ppo/training_log.json")

if __name__ == "__main__":
    all_results = {}
    for ticker in TICKERS:
        df = pd.read_csv(f"data/raw/{ticker}.csv", parse_dates=["date"])
        train_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)

        t0 = time.time()
        result = train_finrl_ppo(train_df, ticker, total_timesteps=TOTAL_TIMESTEPS)
        elapsed = time.time() - t0

        print(f"=== {ticker} ({len(train_df)} train rows, {elapsed:.1f}s) ===")
        for point in result["reward_log"]:
            print(f"  timesteps={point['timesteps']:>6} episode_reward={point['episode_reward']:.4f}")

        all_results[ticker] = {"elapsed_sec": elapsed, "reward_log": result["reward_log"]}

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nTraining log written to {LOG_PATH}")
