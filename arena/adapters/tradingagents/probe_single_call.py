"""One-shot cost/latency probe for a single TradingAgentsGraph.propagate() call.
Hard-gated: this script makes EXACTLY one real LLM-backed decision and stops. Do not
loop this over multiple dates without reviewing the measured cost first.
"""
import json
import time
from pathlib import Path

from langchain_community.callbacks import get_openai_callback

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

STATE_DIR = Path(__file__).resolve().parent / ".state"

PRICE_PER_M = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 1.74, "output": 3.48},
}

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "deepseek"
config["deep_think_llm"] = "deepseek-v4-pro"
config["quick_think_llm"] = "deepseek-v4-flash"
config["memory_log_path"] = str(STATE_DIR / "memory" / "trading_memory.md")
config["data_cache_dir"] = str(STATE_DIR / "cache")
config["results_dir"] = str(STATE_DIR / "logs")

ta = TradingAgentsGraph(debug=True, config=config)

t0 = time.time()
with get_openai_callback() as cb:
    state, decision = ta.propagate("AAPL", "2023-01-03")
elapsed = time.time() - t0

print("=== TIMING ===")
print(f"elapsed_sec={elapsed:.1f}")

print("\n=== LANGCHAIN CALLBACK (aggregate, may not split by model) ===")
print(f"total_tokens={cb.total_tokens} prompt_tokens={cb.prompt_tokens} "
      f"completion_tokens={cb.completion_tokens} successful_requests={cb.successful_requests}")
print(f"langchain_estimated_cost_usd={cb.total_cost}  (OpenAI pricing table -- NOT accurate for DeepSeek, ignore the dollar figure, only the token/request counts are useful)")

print("\n=== DECISION ===")
print(repr(decision)[:2000])

print("\n=== STATE OBJECT SHAPE ===")
if isinstance(state, dict):
    for k, v in state.items():
        v_repr = repr(v)
        print(f"  {k}: {type(v).__name__} ({len(v_repr)} chars repr) -- {v_repr[:150]}")
else:
    print(f"state is type {type(state)}: {repr(state)[:500]}")

result = {
    "elapsed_sec": elapsed,
    "successful_requests": cb.successful_requests,
    "total_tokens": cb.total_tokens,
    "prompt_tokens": cb.prompt_tokens,
    "completion_tokens": cb.completion_tokens,
    "decision": str(decision),
    "state_keys": list(state.keys()) if isinstance(state, dict) else None,
}
(STATE_DIR / "probe_result.json").write_text(json.dumps(result, indent=2, default=str))
print(f"\nProbe result written to {STATE_DIR / 'probe_result.json'}")
