"""Majority-vote ensemble combining the RL (FinRL PPO) and LLM (TradingAgents) signals.

Unanimity rule for two voters: both must agree on BUY (or SELL) to act; any
disagreement -- including either side voting HOLD -- resolves to HOLD.
"""
from agentfusion.agents.finrl_ppo import FinRLPPOAgent
from agentfusion.agents.trading_agents import TradingAgentsAgent
from agentfusion.base import Action, BaseAgent, Signal
from agentfusion.registry import OptimizerRegistry


@OptimizerRegistry.register("majority_vote_ensemble")
class MajorityVoteEnsemble(BaseAgent):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.rl_agent = FinRLPPOAgent(ticker=ticker)
        self.llm_agent = TradingAgentsAgent(ticker=ticker)

    def decide(self, obs: dict) -> Signal:
        rl_signal = self.rl_agent.decide(obs)
        llm_signal = self.llm_agent.decide(obs)

        if rl_signal.action == Action.BUY and llm_signal.action == Action.BUY:
            return Signal(Action.BUY, confidence=min(rl_signal.confidence, llm_signal.confidence))
        if rl_signal.action == Action.SELL and llm_signal.action == Action.SELL:
            return Signal(Action.SELL, confidence=min(rl_signal.confidence, llm_signal.confidence))
        return Signal(Action.HOLD, confidence=0.5)
