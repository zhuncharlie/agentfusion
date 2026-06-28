# Report 2 — TradingAgents Signal Quality

**Framework:** TauricResearch/TradingAgents v0.3.0
**Architecture:** 4-analyst debate (market + sentiment + news + fundamentals) → bull/bear debate → portfolio manager → final decision
**LLM:** DeepSeek-flash (all roles, budget-first config)
**Decisions available:** 317

## Signal Distribution Per Ticker

| Ticker | Days | BUY | HOLD | SELL | Avg elapsed (s) | Avg cost ($) |
|--------|------|-----|------|------|-----------------|-------------|
| AAPL | 64 | 2 | 21 | 41 | 441 | 3.9909 |
| MSFT | 64 | 11 | 39 | 14 | 432 | 4.0234 |
| NVDA | 63 | 26 | 28 | 9 | 501 | 4.0668 |
| GOOGL | 63 | 18 | 32 | 13 | 574 | 4.1432 |
| AMZN | 63 | 29 | 20 | 14 | 421 | 4.0384 |

## Cost & Latency Summary

| Metric | Value |
|--------|-------|
| Total decisions | 317 |
| Total LLM calls | 871,543 (~2749/decision) |
| Total tokens | 7,843,777,874 |
| Total cost (flash-only) | $1284.5669 |
| Avg cost/decision | $4.0523 |
| Avg wall-clock/decision | 473s (7.9 min) |

## Sample Decisions

### Example BUY signal

**AAPL 2024-01-10** → Overweight (BUY)
> ## Final Trading Decision: **Overweight** ### Synthesis of the Risk Analysts’ Debate The debate exposes the core tension: a structurally superior business at a full valuation, caught in a fragile technical recovery with no institutional confirmation. - **The Aggressive Analyst** correctly identifies the long‑term foundation – price above the 200‑day SMA ($178.06), a golden cross intact, and a…

### Example SELL signal

**AAPL 2024-01-01** → Underweight (SELL)
> **Final Trading Decision: Underweight** **Rationale** The debate correctly identifies a tension between Apple’s resilient long-term structure (fortress balance sheet, $100B FCF, 2B+ active devices, golden cross) and deteriorating short-term momentum (MACD bearish crossover, price below 10 EMA and Bollinger middle band, 58% histogram collapse in 10 sessions). The risk/reward is nearly 1:1—3.3%…

### Example HOLD signal

**AAPL 2024-01-09** → Hold (HOLD)
> ## Final Trading Decision: **Hold** ### Synthesis of the Risk Analysts’ Debate The debate among the three analysts has sharpened a clear, balanced picture. No side delivers a knockout argument, and each overreaches in at least one critical area. - **Aggressive Analyst** correctly identifies that the long-term structure (Golden Cross intact, 200‑day SMA at $177.93 providing a 2.9% cushion) and…

## Latest Decisions (most recent 15)

| Ticker | Date | Signal | Raw | Elapsed (s) |
|--------|------|--------|-----|-------------|
| NVDA | 2024-03-07 | HOLD | Hold | 434 |
| NVDA | 2024-03-08 | SELL | Underweight | 490 |
| NVDA | 2024-03-11 | BUY | Overweight | 471 |
| NVDA | 2024-03-12 | SELL | Underweight | 463 |
| NVDA | 2024-03-13 | BUY | Overweight | 518 |
| NVDA | 2024-03-14 | HOLD | Hold | 488 |
| NVDA | 2024-03-15 | HOLD | Hold | 475 |
| NVDA | 2024-03-18 | BUY | Overweight | 525 |
| NVDA | 2024-03-19 | BUY | Overweight | 388 |
| NVDA | 2024-03-20 | HOLD | Hold | 468 |
| NVDA | 2024-03-21 | BUY | Overweight | 446 |
| NVDA | 2024-03-22 | BUY | Overweight | 464 |
| NVDA | 2024-03-25 | SELL | Underweight | 445 |
| NVDA | 2024-03-26 | HOLD | Hold | 440 |
| NVDA | 2024-03-27 | SELL | Underweight | 433 |

_Adapter notes: TradingAgents full run: 315 (ticker,date) pairs (305 NYSE trading days + 10 holiday dates from original Mon-Fri batch), all DeepSeek-flash, ~18 LLM calls per pair._