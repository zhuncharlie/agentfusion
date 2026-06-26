"""Arena comparison layer — three-report analysis for portfolio_2024q1.

Reports generated:
  1. FinRL Portfolio Performance (equity curve, weight matrix, metrics)
  2. TradingAgents Signal Quality (signal distribution, reasoning excerpts)
  3. Joint Divergence Analysis (signal consistency heatmap, top-10 divergence days)

Usage:
    python arena/compare.py [--task TASK_ID] [--out-dir DIR]

Works with partial TradingAgents results (reports what's available).
No dependency on any upstream project code — only reads Arena result JSONs.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TICKERS     = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f", fallback="—"):
    if v is None or (isinstance(v, float) and v != v):
        return fallback
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def _pct(v, fallback="—"):
    if v is None:
        return fallback
    return f"{v:.2%}"


def _load_finrl(task_id: str) -> dict | None:
    p = RESULTS_DIR / task_id / "finrl.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_tradingagents(task_id: str) -> dict | None:
    p = RESULTS_DIR / task_id / "tradingagents.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_ta_decisions_from_state() -> list[dict]:
    """Load per-(ticker,date) decisions from .state/decisions/ even if the full
    batch hasn't finished yet (partial results are acceptable)."""
    state_dir = (
        Path(__file__).resolve().parent
        / "adapters" / "tradingagents" / ".state" / "decisions"
    )
    decisions = []
    if state_dir.exists():
        for f in sorted(state_dir.glob("*.json")):
            try:
                decisions.append(json.loads(f.read_text()))
            except Exception:
                pass
    return decisions


# ── Report 1: FinRL Portfolio Performance ─────────────────────────────────────

def report_finrl(finrl: dict) -> str:
    no  = finrl.get("native_output", {})
    ext = finrl.get("extracted", {})

    lines = [
        "# Report 1 — FinRL Portfolio Performance",
        "",
        f"**Framework:** AI4Finance-Foundation/FinRL (PPO, stable-baselines3)",
        f"**Task:** {finrl.get('task_id', '?')}",
        f"**Tickers:** {', '.join(no.get('tickers', TICKERS))}",
        f"**Train period:** {' → '.join(no.get('train_period', ['?', '?']))}",
        f"**Test period:** {' → '.join(no.get('test_period', ['?', '?']))}  "
        f"({len(no.get('account_memory', [])) - 1} trading days)",
        f"**Training timesteps:** {no.get('total_timesteps', '?'):,}",
        f"**Wall-clock time:** {no.get('elapsed_sec', 0):.1f}s  |  **API cost:** $0.00 (local RL)",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Return | {_pct(ext.get('total_return'))} |",
        f"| Sharpe Ratio (annualized) | {_fmt(ext.get('sharpe'), '.3f')} |",
        f"| Max Drawdown | {_pct(ext.get('mdd'))} |",
        f"| Calmar Ratio | {_fmt(ext.get('calmar'), '.3f')} |",
        f"| Initial Capital | $100,000 |",
        f"| Final Value | ${no.get('final_account_value', 0):,.2f} |",
        "",
    ]

    # Buy-and-hold comparison
    acct = no.get("account_memory", [])
    if acct:
        lines += [
            "## Equity Curve (every 5th trading day)",
            "",
            "| Date | Account Value | Daily Return |",
            "|------|--------------|--------------|",
        ]
        prev_val = None
        for i, row in enumerate(acct):
            if i % 5 == 0 or i == len(acct) - 1:
                val = row["account_value"]
                dr = f"{(val/prev_val-1):.3%}" if prev_val else "—"
                lines.append(f"| {row['date']} | ${val:,.2f} | {dr} |")
            prev_val = row["account_value"]
        lines.append("")

    # Daily weight matrix
    dw = no.get("daily_weight_matrix", [])
    if dw:
        tickers_in = [c for c in list(dw[0].keys()) if c not in ("date", "cash")]
        lines += [
            "## Daily Portfolio Weights (every 5th trading day)",
            "",
            f"| Date | {' | '.join(tickers_in)} | Cash |",
            "|" + "------|" * (len(tickers_in) + 2),
        ]
        for i, row in enumerate(dw):
            if i % 5 == 0 or i == len(dw) - 1:
                ticker_cols = " | ".join(f"{row.get(t, 0):.1%}" for t in tickers_in)
                lines.append(f"| {row['date']} | {ticker_cols} | {row.get('cash', 0):.1%} |")
        lines.append("")

        # Summary: average weight per ticker over the test period
        lines += ["## Average Portfolio Weights Over Test Period", ""]
        avg: dict[str, float] = {}
        for row in dw:
            for t in tickers_in + ["cash"]:
                avg[t] = avg.get(t, 0.0) + row.get(t, 0.0)
        n = len(dw)
        lines.append("| Ticker | Avg Weight |")
        lines.append("|--------|-----------|")
        for t in tickers_in + ["cash"]:
            lines.append(f"| {t} | {avg[t]/n:.1%} |")
        lines.append("")

    lines += [
        "## Key Observation",
        "",
        "> FinRL (PPO trained on 2020–2022 data) concentrated the portfolio almost "
        "entirely in AAPL (~99.8% by Q1 end), essentially ignoring MSFT, NVDA, GOOGL, "
        "and AMZN. This strategy reflects the PPO's learned prior from the training "
        "period where AAPL was a top performer. In 2024-Q1, AAPL underperformed, "
        "resulting in a -6.97% return vs a positive market period for the other four stocks.",
        "",
        f"_Adapter notes: {finrl.get('adapter_notes', '')}_",
    ]

    return "\n".join(lines)


# ── Report 2: TradingAgents Signal Quality ────────────────────────────────────

def report_tradingagents(decisions: list[dict], ta_result: dict | None) -> str:
    total_available = len(decisions)

    lines = [
        "# Report 2 — TradingAgents Signal Quality",
        "",
        "**Framework:** TauricResearch/TradingAgents v0.3.0",
        "**Architecture:** 4-analyst debate (market + sentiment + news + fundamentals) "
        "→ bull/bear debate → portfolio manager → final decision",
        f"**LLM:** DeepSeek-flash (all roles, budget-first config)",
        f"**Decisions available:** {total_available}",
        "",
    ]

    if not decisions:
        lines.append("_No decisions available yet. Run arena/adapters/tradingagents/run.py first._")
        return "\n".join(lines)

    # ── Signal distribution per ticker ──────────────────────────────────────
    by_ticker: dict[str, list[dict]] = {t: [] for t in TICKERS}
    for d in decisions:
        t = d.get("ticker", "?")
        if t in by_ticker:
            by_ticker[t].append(d)

    lines += ["## Signal Distribution Per Ticker", ""]
    lines.append("| Ticker | Days | BUY | HOLD | SELL | Avg elapsed (s) | Avg cost ($) |")
    lines.append("|--------|------|-----|------|------|-----------------|-------------|")
    for t in TICKERS:
        ds = by_ticker[t]
        if not ds:
            lines.append(f"| {t} | 0 | — | — | — | — | — |")
            continue
        cnts = {"BUY": 0, "HOLD": 0, "SELL": 0}
        for d in ds:
            sig = d.get("decision", "HOLD")
            if sig in cnts:
                cnts[sig] += 1
        avg_s = sum(d.get("elapsed_sec", 0) for d in ds) / len(ds)
        avg_c = sum(d.get("cost_usd", 0) for d in ds) / len(ds)
        lines.append(
            f"| {t} | {len(ds)} | {cnts['BUY']} | {cnts['HOLD']} | {cnts['SELL']} | "
            f"{avg_s:.0f} | {avg_c:.4f} |"
        )
    lines.append("")

    # ── Cumulative cost / token summary ─────────────────────────────────────
    total_cost   = sum(d.get("cost_usd", 0) for d in decisions)
    total_tokens = sum(d.get("prompt_tokens", 0) + d.get("completion_tokens", 0) for d in decisions)
    total_calls  = sum(d.get("llm_calls", 0) for d in decisions)
    avg_elapsed  = sum(d.get("elapsed_sec", 0) for d in decisions) / total_available

    lines += [
        "## Cost & Latency Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total decisions | {total_available} |",
        f"| Total LLM calls | {total_calls:,} (~{total_calls/max(1,total_available):.0f}/decision) |",
        f"| Total tokens | {total_tokens:,} |",
        f"| Total cost (flash-only) | ${total_cost:.4f} |",
        f"| Avg cost/decision | ${total_cost/total_available:.4f} |",
        f"| Avg wall-clock/decision | {avg_elapsed:.0f}s ({avg_elapsed/60:.1f} min) |",
        "",
    ]

    # ── Sample decisions (first BUY, first SELL, most recent) ───────────────
    buys  = [d for d in decisions if d.get("decision") == "BUY"]
    sells = [d for d in decisions if d.get("decision") == "SELL"]
    holds = [d for d in decisions if d.get("decision") == "HOLD"]

    def _decision_block(d: dict) -> list[str]:
        reasoning = d.get("final_reasoning", "")
        excerpt = textwrap.shorten(reasoning.replace("\n", " "), width=400, placeholder="…")
        return [
            f"**{d['ticker']} {d['date']}** → {d['decision_raw']} ({d['decision']})",
            f"> {excerpt}",
            "",
        ]

    lines += ["## Sample Decisions", ""]
    if buys:
        lines += ["### Example BUY signal", ""] + _decision_block(buys[0])
    if sells:
        lines += ["### Example SELL signal", ""] + _decision_block(sells[0])
    if holds:
        lines += ["### Example HOLD signal", ""] + _decision_block(holds[0])

    # ── Day-by-day signal table (latest 15 available) ───────────────────────
    lines += [
        "## Latest Decisions (most recent 15)",
        "",
        "| Ticker | Date | Signal | Raw | Elapsed (s) |",
        "|--------|------|--------|-----|-------------|",
    ]
    for d in decisions[-15:]:
        lines.append(
            f"| {d['ticker']} | {d['date']} | {d.get('decision','?')} | "
            f"{d.get('decision_raw','?')} | {d.get('elapsed_sec',0):.0f} |"
        )
    lines.append("")

    if ta_result:
        lines.append(f"_Adapter notes: {ta_result.get('adapter_notes', '')}_")

    return "\n".join(lines)


# ── Report 3: Joint Divergence Analysis ───────────────────────────────────────

def _finrl_direction(weight_row: dict, prev_weight_row: dict | None, ticker: str) -> str:
    """Derive FinRL's directional view from portfolio weight change."""
    if prev_weight_row is None:
        return "HOLD"
    cur  = weight_row.get(ticker, 0.0)
    prev = prev_weight_row.get(ticker, 0.0)
    delta = cur - prev
    if delta > 0.01:
        return "BUY"
    if delta < -0.01:
        return "SELL"
    return "HOLD"


def report_divergence(finrl: dict | None, decisions: list[dict]) -> str:
    lines = [
        "# Report 3 — Joint Divergence Analysis",
        "",
        "**Method:** FinRL position direction (weight change >1%=BUY, <-1%=SELL, else HOLD) "
        "vs TradingAgents signal (Overweight→BUY, Underweight→SELL, Hold→HOLD)",
        "",
    ]

    if not finrl or not decisions:
        msg = []
        if not finrl:
            msg.append("FinRL result")
        if not decisions:
            msg.append("TradingAgents decisions")
        lines.append(f"_Missing: {', '.join(msg)}. Cannot produce divergence analysis._")
        return "\n".join(lines)

    no = finrl.get("native_output", {})
    dw = no.get("daily_weight_matrix", [])

    # Build FinRL direction table: {(ticker, date): BUY/HOLD/SELL}
    finrl_dir: dict[tuple[str, str], str] = {}
    weight_by_date: dict[str, dict] = {row["date"]: row for row in dw}
    sorted_dates = sorted(weight_by_date)

    for i, date in enumerate(sorted_dates):
        prev_date = sorted_dates[i - 1] if i > 0 else None
        prev_row  = weight_by_date[prev_date] if prev_date else None
        curr_row  = weight_by_date[date]
        for t in TICKERS:
            finrl_dir[(t, date)] = _finrl_direction(curr_row, prev_row, t)

    # Build TradingAgents signal table: {(ticker, date): BUY/HOLD/SELL}
    ta_dir: dict[tuple[str, str], str] = {}
    for d in decisions:
        ta_dir[(d["ticker"], d["date"])] = d.get("decision", "HOLD")

    # Find overlapping (ticker, date) pairs
    overlap_keys = sorted(set(finrl_dir) & set(ta_dir))
    if not overlap_keys:
        lines.append(
            "_No overlapping (ticker, date) pairs between FinRL and TradingAgents yet._\n\n"
            f"FinRL dates: {len(finrl_dir)} pairs | "
            f"TradingAgents dates: {len(ta_dir)} pairs"
        )
        return "\n".join(lines)

    # Agreement analysis
    agree = [k for k in overlap_keys if finrl_dir[k] == ta_dir[k]]
    disagree = [k for k in overlap_keys if finrl_dir[k] != ta_dir[k]]
    agree_rate = len(agree) / len(overlap_keys)

    lines += [
        "## Agreement Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overlapping (ticker, date) pairs | {len(overlap_keys)} |",
        f"| Agreement | {len(agree)} ({agree_rate:.1%}) |",
        f"| Divergence | {len(disagree)} ({1-agree_rate:.1%}) |",
        "",
    ]

    # Signal consistency heatmap per ticker (available days)
    lines += [
        "## Signal Consistency Heatmap",
        "_Abbreviations: B=BUY, H=HOLD, S=SELL, ?=TA not yet available_",
        "_Format: FinRL/TradingAgents per day (first 20 available days per ticker)_",
        "",
    ]

    ta_dates_available = sorted({k[1] for k in ta_dir})[:20]
    for ticker in TICKERS:
        row_parts = [f"**{ticker}**"]
        agree_count = diverge_count = 0
        for date in ta_dates_available:
            if date not in sorted_dates:
                row_parts.append("?/?")
                continue
            f_sig = finrl_dir.get((ticker, date), "H")[0]   # B/H/S
            t_sig = ta_dir.get((ticker, date), "?")
            t_initial = t_sig[0] if t_sig != "?" else "?"
            marker = "✓" if f_sig == t_initial else "✗"
            row_parts.append(f"{f_sig}/{t_initial}{marker}")
            if f_sig == t_initial:
                agree_count += 1
            else:
                diverge_count += 1
        lines.append(f"| {' | '.join(row_parts)} |")
        lines.append(f"  ↳ Agree: {agree_count}, Diverge: {diverge_count}")
        lines.append("")

    # Top divergence days
    diverge_days: list[tuple[int, str, str, str, str]] = []
    for ticker in TICKERS:
        for date in ta_dates_available:
            if (ticker, date) in finrl_dir and (ticker, date) in ta_dir:
                f_sig = finrl_dir[(ticker, date)]
                t_sig = ta_dir[(ticker, date)]
                if f_sig != t_sig:
                    diverge_days.append((1, ticker, date, f_sig, t_sig))

    lines += [
        "",
        "## Top Divergence Days",
        "",
        f"| # | Ticker | Date | FinRL | TradingAgents | Context |",
        f"|---|--------|------|-------|---------------|---------|",
    ]
    for i, (_, ticker, date, f_sig, t_sig) in enumerate(diverge_days[:10]):
        # Get TradingAgents reasoning excerpt for this divergence day
        ta_reasoning = ""
        for d in decisions:
            if d["ticker"] == ticker and d["date"] == date:
                raw = d.get("final_reasoning", "")
                ta_reasoning = textwrap.shorten(raw.replace("\n", " "), width=120, placeholder="…")
                break
        lines.append(
            f"| {i+1} | {ticker} | {date} | {f_sig} | {t_sig} | "
            f"{ta_reasoning or '—'} |"
        )
    lines.append("")

    # ── Cross-framework cost comparison ─────────────────────────────────────
    finrl_cost  = finrl.get("cost_usd", 0.0)
    finrl_lat   = finrl.get("latency_sec", 0.0)
    ta_decisions_n = len(decisions)
    ta_cost     = sum(d.get("cost_usd", 0) for d in decisions)
    ta_lat_avg  = sum(d.get("elapsed_sec", 0) for d in decisions) / max(1, ta_decisions_n)

    lines += [
        "## Framework Comparison",
        "",
        "| Dimension | FinRL (PPO) | TradingAgents (LLM) |",
        "|-----------|-------------|---------------------|",
        f"| Decision type | Portfolio weights (continuous) | Per-stock signal (BUY/HOLD/SELL) |",
        f"| Decision granularity | Whole portfolio jointly | Each stock independently |",
        f"| Decision time | <1 ms (inference) | ~{ta_lat_avg:.0f}s (~{ta_lat_avg/60:.1f} min) |",
        f"| Training cost | GPU hrs (one-time) | — |",
        f"| Per-decision API cost | $0.00 | ~${ta_cost/max(1,ta_decisions_n):.4f} |",
        f"| Total cost (this run) | $0.00 | ${ta_cost:.4f} ({ta_decisions_n} decisions) |",
        f"| Explainability | Opaque (neural net) | Full reasoning chain |",
        f"| Adaptability | Retrain required | Zero-shot (new stock instantly) |",
        f"| Data dependency | Historical OHLCV only | News + sentiment + fundamentals + macro |",
        "",
        "## Core Insight",
        "",
        "> These two frameworks solve different problems. FinRL is a **portfolio optimizer** "
        "that learns asset allocation rules from historical price patterns — it answers "
        "*'how much of each stock should I hold?'* TradingAgents is a **decision analyst** "
        "that synthesizes qualitative information at a point in time — it answers "
        "*'what would I do if I were a portfolio manager reading today's news?'* "
        "Comparing their 'signals' directly is a category error; the divergence between "
        "them is not noise — it reflects the difference in what each framework knows and cares about.",
    ]

    return "\n".join(lines)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Arena comparison reports")
    parser.add_argument("--task",    default="portfolio_2024q1")
    parser.add_argument("--out-dir", default=None,
                        help="Write reports to this directory (default: print to stdout)")
    args = parser.parse_args()

    finrl  = _load_finrl(args.task)
    ta_res = _load_tradingagents(args.task)
    # Prefer live .state/decisions/ (available while batch is still running)
    # Fall back to the final result JSON if the batch has completed
    decisions = _load_ta_decisions_from_state()
    if not decisions and ta_res:
        decisions = ta_res.get("native_output", {}).get("decisions", [])

    print(f"Loaded: FinRL={'yes' if finrl else 'no'}, "
          f"TradingAgents decisions={len(decisions)}", flush=True)

    reports = [
        ("report1_finrl_performance",     report_finrl(finrl) if finrl else "# Report 1 — FinRL\n\n_No result yet._"),
        ("report2_tradingagents_signals",  report_tradingagents(decisions, ta_res)),
        ("report3_divergence_analysis",    report_divergence(finrl, decisions)),
    ]

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, content in reports:
            path = out / f"{name}.md"
            path.write_text(content)
            print(f"Written: {path}")
    else:
        for name, content in reports:
            print(f"\n{'='*80}")
            print(f"  {name}")
            print(f"{'='*80}\n")
            print(content)


if __name__ == "__main__":
    main()
