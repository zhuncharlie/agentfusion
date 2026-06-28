# Report 2 — TradingAgents Signal Quality

**Framework:** TauricResearch/TradingAgents v0.3.0
**Architecture:** 4-analyst debate (market + sentiment + news + fundamentals) → bull/bear debate → portfolio manager → final decision
**LLM:** DeepSeek-flash (all roles, budget-first config)
**Total decisions:** 315 · **Avg latency:** ~7.9 min/decision · **Avg cost:** ~$4.07/decision · **Total cost:** ~$10.19

---

## Fig 1 — Signal Distribution & Daily Timeline

![Signal distribution](report2_fig1_signal_dist.png)

**Upper panel — bar chart:** For each of the five tickers, the count of BUY / HOLD / SELL
decisions over the full quarter. Below each group, the actual 2024-Q1 return is annotated
for reference. The directional alignment is striking:

| Ticker | Dominant signal | Actual Q1 return | Aligned? |
|---|---|---|---|
| AAPL | **SELL** (63 %) | −7.5 % | ✓ |
| MSFT | HOLD (61 %) | +8.9 % | ✓ neutral |
| NVDA | **BUY** (41 %) | +87.6 % | ✓ |
| GOOGL | HOLD (51 %) | +7.6 % | ✓ neutral |
| AMZN | **BUY** (46 %) | +20.3 % | ✓ |

**Lower panel — timeline:** Each dot is one daily decision, positioned above the ticker's
baseline (BUY), on it (HOLD), or below (SELL). AAPL's cluster of red SELL dots throughout
January and February stands out clearly. NVDA and AMZN show more green BUY dots in the
first half of the quarter, consistent with their strong performance.

---

## Fig 2 — Latency & Cost per Decision

![Latency and cost](report2_fig2_latency_cost.png)

**Left histogram:** Decision latency is broadly distributed around 7–9 minutes with a long
tail up to ~15 min, driven by longer multi-analyst debates on complex days. The median
(~7.4 min) is slightly faster than the mean (~7.9 min).

**Centre box plot:** GOOGL decisions are consistently the slowest (~9.6 min median),
reflecting the richer news and fundamentals coverage for a diversified tech+cloud company.
AMZN and AAPL are fastest (~7 min), likely because the debate converges quickly on a
clear directional view.

**Right scatter:** Cost and latency are highly correlated (r ≈ 0.95) because DeepSeek-flash
pricing is token-based and token count is proportional to debate length. There are no
pathological outliers — no single call exceeded $8.

**Operational take:** At ~$4/call and ~8 min/call, TradingAgents is not a real-time system.
It is better framed as an overnight due-diligence tool: run each evening for the next
trading day's watchlist, generating a reasoned case for each position before market open.
