"""Single-asset long/flat trading environment with FinRL-style technical-indicator state.

Position semantics mirror agentfusion.backtest.run_backtest exactly (full long or full
flat, same BUY/SELL thresholds) so a policy trained here transfers directly to backtest.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd_diff(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return ((macd - signal) / close).fillna(0.0)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    close = price_df["close"]
    ret = close.pct_change().fillna(0.0)
    features = pd.DataFrame(
        {
            "ret": ret,
            "rsi": _rsi(close) / 100.0,
            "macd_diff": _macd_diff(close),
            "vol_z": _zscore(price_df["volume"], window=20),
            "volatility": ret.rolling(10).std(),
            "momentum_5": ret.rolling(5).mean(),
        }
    )
    return features.fillna(0.0)


class StockTradingEnv(gym.Env):
    """Long/flat single-asset env. Action > BUY_THRESHOLD goes long; < SELL_THRESHOLD goes flat."""

    metadata = {"render_modes": []}

    BUY_THRESHOLD = 0.1
    SELL_THRESHOLD = -0.1

    def __init__(self, price_df: pd.DataFrame, initial_cash: float = 100_000.0, transaction_cost: float = 0.001):
        super().__init__()
        self.price_df = price_df.reset_index(drop=True)
        self.features = compute_features(self.price_df)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        n_features = self.features.shape[1] + 1  # + current position

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

        self._step = 0
        self._position = 0.0
        self._cash = initial_cash
        self._shares = 0.0

    def _obs(self) -> np.ndarray:
        feat = self.features.iloc[self._step].to_numpy(dtype=np.float32)
        return np.concatenate([feat, [self._position]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._position = 0.0
        self._cash = self.initial_cash
        self._shares = 0.0
        return self._obs(), {}

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        price = self.price_df["close"].iloc[self._step]
        equity_before = self._cash + self._shares * price

        if a > self.BUY_THRESHOLD and self._position == 0.0:
            cost = self._cash * self.transaction_cost
            self._shares = (self._cash - cost) / price
            self._cash = 0.0
            self._position = 1.0
        elif a < self.SELL_THRESHOLD and self._position == 1.0:
            proceeds = self._shares * price
            self._cash += proceeds - proceeds * self.transaction_cost
            self._shares = 0.0
            self._position = 0.0
        # else: hold, no-op

        self._step += 1
        terminated = self._step >= len(self.price_df) - 1
        next_price = self.price_df["close"].iloc[self._step]
        equity_after = self._cash + self._shares * next_price
        reward = (equity_after / equity_before - 1.0) * 100.0

        return self._obs(), reward, terminated, False, {}
