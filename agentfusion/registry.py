"""Central registry so third-party agents can plug in without touching core code."""
from typing import Dict, Type

from agentfusion.base import BaseAgent


class OptimizerRegistry:
    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(agent_cls: Type[BaseAgent]):
            if not issubclass(agent_cls, BaseAgent):
                raise TypeError(f"{agent_cls.__name__} must subclass BaseAgent")
            if name in cls._agents and cls._agents[name] is not agent_cls:
                raise ValueError(f"Agent name '{name}' is already registered to {cls._agents[name].__name__}")
            cls._agents[name] = agent_cls
            return agent_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseAgent]:
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not registered. Available: {cls.list()}")
        return cls._agents[name]

    @classmethod
    def list(cls):
        return sorted(cls._agents.keys())
