"""Canonical Buy-and-Hold baseline agent, reused across experiment-pipeline scripts."""
from agentfusion.base import Action, BaseAgent, Signal
from agentfusion.registry import OptimizerRegistry


@OptimizerRegistry.register("buy_hold")
class BuyHoldAgent(BaseAgent):
    def decide(self, obs: dict) -> Signal:
        is_first_bar = len(obs["history"]) == 1
        return Signal(Action.BUY if is_first_bar else Action.HOLD)
