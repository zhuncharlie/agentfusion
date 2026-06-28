# Report 3 — Joint Divergence Analysis

**Method:** FinRL position direction (weight change >1%=BUY, <-1%=SELL, else HOLD) vs TradingAgents signal (Overweight→BUY, Underweight→SELL, Hold→HOLD)

## Agreement Summary

| Metric | Value |
|--------|-------|
| Overlapping (ticker, date) pairs | 300 |
| Agreement | 130 (43.3%) |
| Divergence | 170 (56.7%) |

## Signal Consistency Heatmap
_Abbreviations: B=BUY, H=HOLD, S=SELL, ?=TA not yet available_
_Format: FinRL/TradingAgents per day (first 20 available days per ticker)_

| **AAPL** | ?/? | H/S✗ | B/S✗ | B/S✗ | B/S✗ | B/S✗ | B/H✗ | B/B✓ | B/S✗ | S/H✗ | ?/? | H/S✗ | B/S✗ | B/H✗ | H/S✗ | H/S✗ | B/B✓ | S/S✓ | S/S✓ | H/H✓ |
  ↳ Agree: 5, Diverge: 13

| **MSFT** | ?/? | H/S✗ | H/H✓ | H/S✗ | H/H✓ | H/S✗ | H/S✗ | H/S✗ | H/H✓ | H/H✓ | ?/? | H/H✓ | H/H✓ | H/S✗ | H/H✓ | H/B✗ | H/H✓ | H/S✗ | H/H✓ | S/S✓ |
  ↳ Agree: 10, Diverge: 8

| **NVDA** | ?/? | H/H✓ | H/H✓ | H/H✓ | H/B✗ | H/B✗ | H/B✗ | H/H✓ | H/B✗ | H/H✓ | ?/? | H/H✓ | H/H✓ | H/S✗ | H/H✓ | H/H✓ | H/H✓ | H/B✗ | H/S✗ | H/H✓ |
  ↳ Agree: 11, Diverge: 7

| **GOOGL** | ?/? | H/B✗ | H/H✓ | H/B✗ | H/H✓ | H/B✗ | H/H✓ | H/B✗ | H/H✓ | H/H✓ | ?/? | H/B✗ | H/B✗ | H/H✓ | H/B✗ | H/B✗ | H/H✓ | H/H✓ | H/S✗ | H/H✓ |
  ↳ Agree: 9, Diverge: 9

| **AMZN** | ?/? | H/B✗ | H/B✗ | H/S✗ | H/H✓ | H/H✓ | H/S✗ | H/B✗ | H/H✓ | H/B✗ | ?/? | H/B✗ | H/B✗ | H/H✓ | H/B✗ | H/S✗ | H/B✗ | H/B✗ | H/H✓ | H/S✗ |
  ↳ Agree: 5, Diverge: 13


## Top Divergence Days

| # | Ticker | Date | FinRL | TradingAgents | Context |
|---|--------|------|-------|---------------|---------|
| 1 | AAPL | 2024-01-02 | HOLD | SELL | **Final Trading Decision: Underweight** After synthesizing the debate, I find the **Neutral Analyst’s** recommendation… |
| 2 | AAPL | 2024-01-03 | BUY | SELL | **Rating**: Underweight **Executive Summary**: Reduce AAPL position by 15–20% on intraday bounces toward the $185–187… |
| 3 | AAPL | 2024-01-04 | BUY | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Debate The risk analysts' debate has sharpened the key… |
| 4 | AAPL | 2024-01-05 | BUY | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Analysts’ Debate The risk analysts have laid out a… |
| 5 | AAPL | 2024-01-08 | BUY | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Analysts’ Debate The three risk analysts have each… |
| 6 | AAPL | 2024-01-09 | BUY | HOLD | ## Final Trading Decision: **Hold** ### Synthesis of the Risk Analysts’ Debate The debate among the three analysts has… |
| 7 | AAPL | 2024-01-11 | BUY | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Risk Analysts’ Debate The debate confirms a clear near-… |
| 8 | AAPL | 2024-01-12 | SELL | HOLD | ## Final Trading Decision: **Hold** ### Synthesis of the Risk Analysts’ Debate The three analysts have presented a… |
| 9 | AAPL | 2024-01-16 | HOLD | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Risk Analysts’ Debate The debate confirms a clear… |
| 10 | AAPL | 2024-01-17 | BUY | SELL | ## Final Trading Decision: **Underweight** ### Synthesis of the Risk Analysts’ Debate The debate exposes a clear… |

## Framework Comparison

| Dimension | FinRL (PPO) | TradingAgents (LLM) |
|-----------|-------------|---------------------|
| Decision type | Portfolio weights (continuous) | Per-stock signal (BUY/HOLD/SELL) |
| Decision granularity | Whole portfolio jointly | Each stock independently |
| Decision time | <1 ms (inference) | ~473s (~7.9 min) |
| Training cost | GPU hrs (one-time) | — |
| Per-decision API cost | $0.00 | ~$4.0523 |
| Total cost (this run) | $0.00 | $1284.5669 (317 decisions) |
| Explainability | Opaque (neural net) | Full reasoning chain |
| Adaptability | Retrain required | Zero-shot (new stock instantly) |
| Data dependency | Historical OHLCV only | News + sentiment + fundamentals + macro |

## Core Insight

> These two frameworks solve different problems. FinRL is a **portfolio optimizer** that learns asset allocation rules from historical price patterns — it answers *'how much of each stock should I hold?'* TradingAgents is a **decision analyst** that synthesizes qualitative information at a point in time — it answers *'what would I do if I were a portfolio manager reading today's news?'* Comparing their 'signals' directly is a category error; the divergence between them is not noise — it reflects the difference in what each framework knows and cares about.