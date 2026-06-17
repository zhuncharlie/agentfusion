"""FinRL-style PPO trading agent: technical-indicator state, long/flat position."""
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agentfusion.base import Action, BaseAgent, Signal
from agentfusion.envs import StockTradingEnv, compute_features
from agentfusion.registry import OptimizerRegistry

CHECKPOINT_DIR = Path("checkpoints/finrl_ppo")


def train_finrl_ppo(
    price_df: pd.DataFrame,
    ticker: str,
    total_timesteps: int = 50_000,
    seed: int = 0,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    """Train PPO on price_df (training-period data only) and save a per-ticker checkpoint."""
    env = DummyVecEnv([lambda: StockTradingEnv(price_df)])
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu")

    reward_log = []
    chunk = max(total_timesteps // 10, 1000)
    steps_done = 0
    while steps_done < total_timesteps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        mean_reward = _evaluate_episode_reward(model, price_df)
        reward_log.append({"timesteps": steps_done, "episode_reward": mean_reward})

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_dir / ticker)
    return {"ticker": ticker, "reward_log": reward_log, "checkpoint": str(checkpoint_dir / f"{ticker}.zip")}


def _evaluate_episode_reward(model, price_df: pd.DataFrame) -> float:
    env = StockTradingEnv(price_df)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


@OptimizerRegistry.register("finrl_ppo")
class FinRLPPOAgent(BaseAgent):
    """Loads a per-ticker PPO checkpoint; discretizes the continuous action into BUY/HOLD/SELL."""

    def __init__(self, ticker: str, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.ticker = ticker
        self.model = PPO.load(Path(checkpoint_dir) / ticker)
        self._position = 0.0

    def decide(self, obs: dict) -> Signal:
        history = obs["history"]
        if len(history) == 1:
            self._position = 0.0

        features = compute_features(history)
        feat_vec = features.iloc[-1].to_numpy(dtype=np.float32)
        model_obs = np.concatenate([feat_vec, [self._position]]).astype(np.float32)

        action, _ = self.model.predict(model_obs, deterministic=True)
        a = float(np.asarray(action).reshape(-1)[0])

        if a > StockTradingEnv.BUY_THRESHOLD and self._position == 0.0:
            self._position = 1.0
            return Signal(Action.BUY, confidence=min(abs(a), 1.0))
        if a < StockTradingEnv.SELL_THRESHOLD and self._position == 1.0:
            self._position = 0.0
            return Signal(Action.SELL, confidence=min(abs(a), 1.0))
        return Signal(Action.HOLD, confidence=1.0 - min(abs(a), 1.0))
