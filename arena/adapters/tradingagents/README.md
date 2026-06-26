# TradingAgents adapter (real upstream package)

Wraps the actual [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
package (v0.3.0 at time of writing) — not a reimplementation.

TradingAgents is a **multi-analyst LLM debate system**: for each (ticker, date), it runs
four specialist analysts (market, sentiment, news, fundamentals) through two structured
debate rounds before a portfolio manager delivers a final decision. This adapter runs the
real `TradingAgentsGraph.propagate()` pipeline end to end and writes results into the
Arena schema (`arena/SCHEMA.md`).

## Setup

**IMPORTANT:** Do NOT `pip install tradingagents`. The PyPI package by that name
(`tradingagents` v0.7.0) is an unrelated project (Mai0313/tradingagents) that has nothing
to do with TauricResearch. It will fail at runtime.

```bash
conda env create -f arena/adapters/tradingagents/env.yml
# If you need to install manually:
conda create -n tradingagents_real python=3.12 rust -c conda-forge -y
conda run -n tradingagents_real pip install tiktoken==0.11.0 langchain-community
conda run -n tradingagents_real pip install git+https://github.com/TauricResearch/TradingAgents.git
```

## Required environment variables

```bash
export DEEPSEEK_API_KEY=sk-...   # Never hardcode; never log this value
```

Optional (graceful degradation if absent):
```bash
export FRED_API_KEY=...          # Macro data; adapter degrades without it
```

## Running the adapter

```bash
# Full run: 5 stocks × 60 trading days (2024-Q1), all DeepSeek-flash
# Estimated cost: $10-15; estimated wall time: 36h serial (runs safely in background)
conda run -n tradingagents_real python arena/adapters/tradingagents/run.py

# Run in background with progress log:
nohup conda run -n tradingagents_real python arena/adapters/tradingagents/run.py \
    > /tmp/ta_run.log 2>&1 & disown
tail -f /tmp/ta_run.log
```

The run is **resumable**: progress is checkpointed to
`arena/adapters/tradingagents/.state/progress.json` after every completed (ticker, date)
pair. Each raw decision is saved to `.state/decisions/{TICKER}_{DATE}.json`. Re-running
the script skips any already-completed pairs automatically.

## Pre-run checklist

1. **Clear memory file** — TradingAgents accumulates reflections across runs:
   ```bash
   > arena/adapters/tradingagents/.state/memory/trading_memory.md
   ```
2. **Confirm DEEPSEEK_API_KEY is set** — the script exits immediately if it's absent.
3. **Estimate cost** — the script prints a lower-bound estimate before starting.

## Real API used (verified against 0.3.0 source)

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["llm_provider"]    = "deepseek"
config["deep_think_llm"]  = "deepseek-v4-flash"
config["quick_think_llm"] = "deepseek-v4-flash"
config["memory_log_path"] = "/path/to/trading_memory.md"
config["data_cache_dir"]  = "/path/to/cache"
config["results_dir"]     = "/path/to/logs"

ta = TradingAgentsGraph(debug=False, config=config)
state, decision = ta.propagate("AAPL", "2024-01-15")

# decision: "Overweight" | "Hold" | "Underweight"  (portfolio-manager vocab)
# state keys: messages, company_of_interest, asset_type, instrument_context,
#             trade_date, market_report, sentiment_report, news_report,
#             fundamentals_report, investment_debate_state, risk_debate_state,
#             past_context, investment_plan, sender, trader_investment_plan,
#             final_trade_decision
```

Decision vocabulary is **portfolio-manager style** (`Overweight`/`Hold`/`Underweight`),
NOT `BUY`/`HOLD`/`SELL`. The adapter maps these to the Arena standard format.

## Measured performance (single-call probe, 2026-06-18)

```
Ticker/date:     AAPL, 2023-01-03
LLM calls:       18
Tokens:          162,821 (131K prompt / 31K completion)
Wall time:       435.9 seconds (7.3 minutes)
Cost (flash):    $0.027
Decision:        Underweight
```

Scaling to 5 tickers × 60 days = 300 calls:
- Cost:      ~$8.1 (flash-only) to $101 (all-pro)
- Wall time: ~36.5 hours serial

## What is and is not stored

**Stored per (ticker, date):**
- Raw decision string (`Overweight`/`Hold`/`Underweight`)
- Mapped decision (`BUY`/`HOLD`/`SELL`)
- One conclusion sentence per analyst (market, sentiment, news, fundamentals)
- First 500 chars of final reasoning
- Token counts and latency

**NOT stored** (to keep size manageable):
- Full analyst reports (can be 10k+ chars each)
- Raw `messages` list
- Full `investment_debate_state` / `risk_debate_state` content

## Gotchas

- **PyPI impostor package**: `pip install tradingagents` installs Mai0313/tradingagents
  (v0.7.0), not TauricResearch. The tell: `from tradingagents.default_config import
  DEFAULT_CONFIG` raises `ModuleNotFoundError`. Fix: uninstall and install from git.
- **tiktoken + Rust**: tiktoken requires Rust ≥ 1.85 to compile from source. The
  conda-forge `rust` package may resolve to an older version. Fix: use
  `tiktoken==0.11.0` which has a prebuilt wheel for cp312-linux.
- **Memory file pollution**: TradingAgents writes per-decision reflections to the memory
  file. Clear it before starting a new experiment; accumulated memory from a previous
  run influences future decisions.
- **Reddit 429 rate limits**: Gracefully degraded by the framework — logs a warning but
  continues without Reddit data. Not an error.
- **FRED_API_KEY absent**: Gracefully degraded — macro indicators fetched from FRED are
  skipped. Not an error.
- **DeepSeek empty response**: Upstream v0.3.0 includes retry logic for this. If you
  see a `NoneType` error on the decision, the retry budget was exceeded.
