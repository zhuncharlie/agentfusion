"""
TradingAgents Arena adapter — 5 stocks × 60 trading days, all DeepSeek-flash.

Budget target: $10-15 total.
Decision cost: ~$0.027/call (flash-only).
Total calls: 300 (5 tickers × ~60 trading days).

Run inside the tradingagents_real conda env:
    conda run -n tradingagents_real python arena/adapters/tradingagents/run.py

Outputs:
    arena/results/portfolio_2024q1/tradingagents.json   — full Arena result
    arena/adapters/tradingagents/.state/progress.json   — resumable checkpoint
    arena/adapters/tradingagents/.state/decisions/      — per-(ticker,date) raw state

Security: DEEPSEEK_API_KEY read from environment only, never logged or written to files.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]
STATE_DIR = Path(__file__).resolve().parent / ".state"
DECISIONS_DIR = STATE_DIR / "decisions"
PROGRESS_FILE = STATE_DIR / "progress.json"
RESULT_FILE = REPO / "arena" / "results" / "portfolio_2024q1" / "tradingagents.json"

for d in [STATE_DIR, DECISIONS_DIR, STATE_DIR / "memory", STATE_DIR / "cache",
          STATE_DIR / "logs", RESULT_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

# ── experiment parameters ──────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
TEST_START = "2024-01-01"
TEST_END   = "2024-03-31"

# ── LLM config — all flash to stay inside $10-15 budget ───────────────────────
TA_CONFIG = {
    "llm_provider":    "deepseek",
    "deep_think_llm":  "deepseek-v4-flash",
    "quick_think_llm": "deepseek-v4-flash",
    "memory_log_path": str(STATE_DIR / "memory" / "trading_memory.md"),
    "data_cache_dir":  str(STATE_DIR / "cache"),
    "results_dir":     str(STATE_DIR / "logs"),
}

# ── cost tracking constants (flash pricing per million tokens) ─────────────────
FLASH_INPUT_PER_M  = 0.14
FLASH_OUTPUT_PER_M = 0.28
COST_HARD_LIMIT_USD = 15.0   # auto-stop when cumulative cost exceeds this


def trading_days(start: str, end: str) -> list[str]:
    """Return actual NYSE trading days between start and end inclusive.

    Uses the OHLCV data already downloaded for AAPL (most liquid, fewest gaps)
    as the ground-truth trading calendar, so TA and FinRL dates align exactly.
    Falls back to Mon-Fri if the CSV is not available.
    """
    csv = Path(__file__).resolve().parents[3] / "data" / "raw" / "AAPL.csv"
    if csv.exists():
        import csv as _csv
        with csv.open() as f:
            reader = _csv.DictReader(f)
            dates = [row["date"][:10] for row in reader
                     if start <= row["date"][:10] <= end]
        return sorted(set(dates))

    # fallback: Mon-Fri
    days = []
    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur <= end_dt:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def decision_path(ticker: str, date: str) -> Path:
    return DECISIONS_DIR / f"{ticker}_{date}.json"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "cost_usd": 0.0, "total_tokens": 0}


def save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))


def extract_analyst_conclusions(state: dict) -> dict[str, str]:
    """Pull one conclusion sentence per analyst from the state object."""
    out: dict[str, str] = {}
    for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report"):
        raw = state.get(key, "") or ""
        # Grab first non-empty sentence ending with period, question mark, or exclamation.
        match = re.search(r"[A-Z][^.!?]*[.!?]", str(raw))
        out[key] = match.group(0).strip() if match else str(raw)[:200].strip()
    return out


def map_decision(raw: str) -> str:
    """Map TradingAgents native vocab to BUY/HOLD/SELL."""
    lower = str(raw).lower()
    if "overweight" in lower or "buy" in lower or "strong buy" in lower:
        return "BUY"
    if "underweight" in lower or "sell" in lower or "strong sell" in lower:
        return "SELL"
    return "HOLD"


def run_one(ta, ticker: str, date: str, cb) -> dict:
    """Run a single propagate() call and return a structured result dict."""
    path = decision_path(ticker, date)
    if path.exists():
        cached = json.loads(path.read_text())
        print(f"  [cached] {ticker} {date} → {cached['decision_raw']}", flush=True)
        return cached

    t0 = time.time()
    state, decision = ta.propagate(ticker, date)
    elapsed = time.time() - t0

    prompt_tokens     = cb.prompt_tokens
    completion_tokens = cb.completion_tokens

    cost_usd = (
        prompt_tokens     / 1_000_000 * FLASH_INPUT_PER_M +
        completion_tokens / 1_000_000 * FLASH_OUTPUT_PER_M
    )

    analyst_conclusions = extract_analyst_conclusions(state)
    final_reasoning = ""
    for key in ("final_trade_decision", "trader_investment_plan", "investment_plan"):
        raw_val = state.get(key) or ""
        if raw_val:
            final_reasoning = str(raw_val)[:500].strip()
            break

    result = {
        "ticker":               ticker,
        "date":                 date,
        "decision_raw":         str(decision),
        "decision":             map_decision(str(decision)),
        "analyst_conclusions":  analyst_conclusions,
        "final_reasoning":      final_reasoning,
        "elapsed_sec":          round(elapsed, 1),
        "prompt_tokens":        prompt_tokens,
        "completion_tokens":    completion_tokens,
        "cost_usd":             round(cost_usd, 4),
        "llm_calls":            cb.successful_requests,
    }

    path.write_text(json.dumps(result, indent=2, default=str))
    return result


def build_final_result(decisions: list[dict], prog: dict, wall_seconds: float) -> dict:
    """Assemble the Arena result schema object."""
    native_output = {
        "decisions":          decisions,
        "framework":          "TradingAgents v0.3.0 (TauricResearch/TradingAgents)",
        "llm_config":         {k: v for k, v in TA_CONFIG.items()
                               if "key" not in k.lower() and "token" not in k.lower()},
        "tickers":            TICKERS,
        "test_start":         TEST_START,
        "test_end":           TEST_END,
        "total_llm_calls":    sum(d.get("llm_calls", 0) for d in decisions),
        "total_tokens":       prog["total_tokens"],
        "total_cost_usd":     round(prog["cost_usd"], 4),
    }

    signal_counts: dict[str, dict] = {t: {"BUY": 0, "HOLD": 0, "SELL": 0} for t in TICKERS}
    for d in decisions:
        ticker = d["ticker"]
        sig    = d.get("decision", "HOLD")
        if ticker in signal_counts and sig in signal_counts[ticker]:
            signal_counts[ticker][sig] += 1

    extracted = {
        "action":           None,
        "confidence":       None,
        "predicted_return": None,
        "sharpe":           None,
        "total_return":     None,
        "mdd":              None,
        "signal_summary":   signal_counts,
    }

    return {
        "project":       "TradingAgents",
        "task_id":       "portfolio_2024q1",
        "native_output": native_output,
        "extracted":     extracted,
        "cost_usd":      round(prog["cost_usd"], 4),
        "latency_sec":   round(wall_seconds, 1),
        "adapter_notes": (
            "TradingAgents runs 18 LLM calls per (ticker, date) through a "
            "multi-analyst debate pipeline (market, sentiment, news, fundamentals + "
            "two debate rounds + portfolio manager). Decision vocabulary is "
            "Overweight/Hold/Underweight (portfolio-manager style), mapped to "
            "BUY/HOLD/SELL in the extracted field. native_output.decisions contains "
            "per-analyst conclusion sentences and final reasoning (first 500 chars). "
            "Full raw state is NOT stored to keep cost/size manageable. "
            "Decisions use all DeepSeek-flash to stay within $10-15 budget."
        ),
    }


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("ERROR: DEEPSEEK_API_KEY environment variable not set")

    from langchain_community.callbacks import get_openai_callback
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config.update(TA_CONFIG)

    print("Initialising TradingAgentsGraph...", flush=True)
    ta = TradingAgentsGraph(debug=False, config=config)

    days   = trading_days(TEST_START, TEST_END)
    pairs  = [(ticker, day) for day in days for ticker in TICKERS]
    prog   = load_progress()
    done   = set(prog["done"])
    remaining = [(t, d) for t, d in pairs if f"{t}:{d}" not in done]

    print(f"Total pairs: {len(pairs)}, already done: {len(done)}, remaining: {len(remaining)}", flush=True)
    print(f"Estimated cost for remaining: ${len(remaining) * 0.027:.2f} (lower bound, flash-only)", flush=True)

    wall_t0   = time.time()
    decisions = []

    # Load already-completed decisions so we can include them in the final output
    for key in sorted(done):
        ticker, date = key.split(":", 1)
        path = decision_path(ticker, date)
        if path.exists():
            decisions.append(json.loads(path.read_text()))

    completed_this_run = 0
    budget_hit = False

    with get_openai_callback() as cb:
        for i, (ticker, date) in enumerate(remaining):
            print(f"[{i+1}/{len(remaining)}] {ticker} {date} ...", flush=True)
            try:
                result = run_one(ta, ticker, date, cb)
                decisions.append(result)
                done.add(f"{ticker}:{date}")
                completed_this_run += 1

                # Update cumulative counters from callback snapshot
                prog["done"]         = sorted(done)
                prog["cost_usd"]     = round(
                    cb.prompt_tokens     / 1_000_000 * FLASH_INPUT_PER_M +
                    cb.completion_tokens / 1_000_000 * FLASH_OUTPUT_PER_M, 4
                )
                prog["total_tokens"] = cb.total_tokens
                save_progress(prog)

                print(
                    f"  → {result['decision_raw']} ({result['decision']}) "
                    f"in {result['elapsed_sec']}s, ~${result['cost_usd']:.4f} | "
                    f"cumulative ${prog['cost_usd']:.3f}",
                    flush=True,
                )

                # Every 10 completed pairs: print a progress summary
                if completed_this_run % 10 == 0:
                    pct = 100 * len(done) / len(pairs)
                    print(
                        f"\n── PROGRESS SUMMARY (every-10 checkpoint) ──────────────────",
                        flush=True,
                    )
                    print(f"  Completed: {len(done)}/{len(pairs)} ({pct:.1f}%)", flush=True)
                    print(f"  Cumulative cost: ${prog['cost_usd']:.4f}", flush=True)
                    print(f"  Total tokens:    {prog['total_tokens']:,}", flush=True)
                    print(f"  Wall time so far: {(time.time()-wall_t0)/3600:.2f}h", flush=True)
                    print(f"────────────────────────────────────────────────────────────\n",
                          flush=True)

                # Hard cost limit
                if prog["cost_usd"] >= COST_HARD_LIMIT_USD:
                    print(
                        f"\n⚠ COST LIMIT REACHED: ${prog['cost_usd']:.4f} ≥ "
                        f"${COST_HARD_LIMIT_USD}. Stopping early.",
                        flush=True,
                    )
                    budget_hit = True
                    break

            except Exception as exc:
                print(f"  ERROR on {ticker} {date}: {exc}", flush=True)
                # Don't mark as done; will retry on next run

    if budget_hit:
        print(f"Run stopped at cost limit. {len(done)}/{len(pairs)} pairs completed.", flush=True)

    wall_seconds = time.time() - wall_t0
    final = build_final_result(decisions, prog, wall_seconds)
    RESULT_FILE.write_text(json.dumps(final, indent=2, default=str))

    print(f"\nDone. {len(decisions)} decisions written to {RESULT_FILE}", flush=True)
    print(f"Total cost: ${prog['cost_usd']:.4f}, tokens: {prog['total_tokens']}, "
          f"wall time: {wall_seconds/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
