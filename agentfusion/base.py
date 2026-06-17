"""Core agent interface that every AgentFusion strategy must implement."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class Signal:
    action: Action
    confidence: float = 1.0


class BaseAgent(ABC):
    """Subclass this and implement `decide` to plug into the Registry/backtest engine.

    `obs` is a dict with keys:
      - ticker: str
      - date: pandas.Timestamp (current bar's date)
      - row: pandas.Series (current bar's OHLCV)
      - history: pandas.DataFrame (all bars up to and including `row`, ascending by date)
    `decide` must only use `obs` — no lookahead into future bars.
    """

    @abstractmethod
    def decide(self, obs: dict) -> Signal:
        raise NotImplementedError
