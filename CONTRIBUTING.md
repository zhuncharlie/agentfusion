# Contributing to AgentFusion

Thanks for considering a contribution. The project is intentionally small right now —
the goal is a clean interface that's easy to extend, not a finished product.

## Adding a new agent

1. Subclass `agentfusion.base.BaseAgent` and implement `decide(self, obs: dict) -> Signal`.
   `obs` contains `ticker`, `date`, `row` (today's OHLCV), and `history` (all bars up to
   and including today, ascending by date). Don't look beyond `history` — that's the
   no-lookahead boundary the backtest engine relies on.
2. Register it with `OptimizerRegistry.register("your_agent_name")` as a class decorator.
3. That's it — your agent now works with `agentfusion.backtest.run_backtest` and shows
   up in `OptimizerRegistry.list()`.

```python
from agentfusion import Action, BaseAgent, OptimizerRegistry, Signal

@OptimizerRegistry.register("my_agent")
class MyAgent(BaseAgent):
    def decide(self, obs): return Signal(Action.HOLD)
```

See `examples/dummy_agent.py` and `examples/community_agent.py` for complete, runnable
examples (registration + backtest).

## Running tests

```bash
pip install -e .[dev]
pytest tests/
```

The sandbox CI (`.github/workflows/ci-sandbox.yml`) runs `examples/dummy_agent.py` and
`examples/community_agent.py` on every PR and just checks they exit cleanly — that's the
bar for "doesn't break the pluggable interface."

## Submitting a PR

- Keep PRs focused: one agent, one fix, or one analysis script per PR.
- If you're adding an agent with heavy dependencies (torch, an LLM SDK, etc.), put them
  in an `optional-dependencies` extra in `pyproject.toml` rather than the base install —
  see how `rl` and `llm` extras are scoped for precedent.
- Label-friendly tasks to start with are tagged `good-first-issue` on the issue tracker;
  see `refine-logs/EXPERIMENT_PLAN.md`'s "社区任务清单" section for the original list
  (bootstrap CI, multi-seed FinRL training, Jobson-Korkie test, expanded stock pool, a
  FinMem adapter, clustered-SE signal-agreement test).

## Results and disclaimers

Anything under `results/` is preliminary unless stated otherwise — it's there to show
the pipeline works end-to-end, not to make a performance claim. If you find something
broken or misleading, please open an issue rather than silently distrusting the numbers.
