# Report 2 — TradingAgents Signal Quality

**Framework:** TauricResearch/TradingAgents v0.3.0
**Architecture:** 4-analyst debate (market + sentiment + news + fundamentals) → bull/bear debate → portfolio manager → final decision
**LLM:** DeepSeek-flash (all roles, budget-first config)
**Decisions available:** 1

## Signal Distribution Per Ticker

| Ticker | Days | BUY | HOLD | SELL | Avg elapsed (s) | Avg cost ($) |
|--------|------|-----|------|------|-----------------|-------------|
| AAPL | 1 | 0 | 0 | 1 | 389 | 0.0321 |
| MSFT | 0 | — | — | — | — | — |
| NVDA | 0 | — | — | — | — | — |
| GOOGL | 0 | — | — | — | — | — |
| AMZN | 0 | — | — | — | — | — |

## Cost & Latency Summary

| Metric | Value |
|--------|-------|
| Total decisions | 1 |
| Total LLM calls | 20 (~20/decision) |
| Total tokens | 196,343 |
| Total cost (flash-only) | $0.0321 |
| Avg cost/decision | $0.0321 |
| Avg wall-clock/decision | 389s (6.5 min) |

## Sample Decisions

### Example SELL signal

**AAPL 2024-01-01** → Underweight (SELL)
> **Final Trading Decision: Underweight** **Rationale** The debate correctly identifies a tension between Apple’s resilient long-term structure (fortress balance sheet, $100B FCF, 2B+ active devices, golden cross) and deteriorating short-term momentum (MACD bearish crossover, price below 10 EMA and Bollinger middle band, 58% histogram collapse in 10 sessions). The risk/reward is nearly 1:1—3.3%…

## Latest Decisions (most recent 15)

| Ticker | Date | Signal | Raw | Elapsed (s) |
|--------|------|--------|-----|-------------|
| AAPL | 2024-01-01 | SELL | Underweight | 389 |
