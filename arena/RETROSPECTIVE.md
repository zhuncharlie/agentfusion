# Arena Pilot Retrospective — 2024-Q1

**Scope:** 5 stocks (AAPL · MSFT · NVDA · GOOGL · AMZN), 2024-Q1 test period,
real FinRL (`StockTradingEnv` + PPO) × real TradingAgents (18-call multi-analyst debate).
**Duration:** ~4 days of compute (FinRL: 2 min; TradingAgents: 41 h).
**Total cost:** $10.19 (TradingAgents, DeepSeek-flash) + $0.00 (FinRL, local GPU).
**Artefacts:** `arena/results/portfolio_2024q1/`, `arena/reports/2024q1/` (7 figures, 3 reports).

---

## 1. What We Set Out to Do

Wire two published frameworks into the AgentFusion `BaseAgent` interface and run them on
the same task side-by-side, producing a quantitative comparison without re-implementing
either framework's methodology. The deliverables were:

1. A working FinRL adapter (`arena/adapters/finrl/run.py`)
2. A working TradingAgents adapter (`arena/adapters/tradingagents/run.py`)
3. A comparison layer (`arena/compare.py`) generating three structured reports
4. This retrospective

---

## 2. What Worked

### FinRL

| Item | Outcome |
|---|---|
| `StockTradingEnv` integration | Clean. State layout documented; `state_space = 1 + 10×stock_dim`. |
| PPO training | Converged in < 3 min on CPU (50 k timesteps). GPU is available but SB3 warns MlpPolicy is slower on GPU than CPU due to small batch sizes. |
| `ConstrainedStockTradingEnv` | Subclassing `step()` to pre-clip buy actions before calling `super()` worked on first attempt. Return flipped from −6.97 % to +0.81 % with zero model changes. |
| Optimal scenario (10 stocks, 2021-H1) | +19.94 % return, Sharpe +1.55. Proved the algorithm is not the bottleneck — scenario design is. |

### TradingAgents

| Item | Outcome |
|---|---|
| Resumable batch | `progress.json` checkpoint + per-call `.state/decisions/` file cache worked correctly. Survived two process crashes (restarted from exact last completed pair). |
| Signal direction accuracy | Matched realised Q1 returns on every ticker: AAPL SELL (−7.5 % ✓), NVDA BUY (+87.6 % ✓), AMZN BUY (+20.3 % ✓). Directional accuracy 72 % vs FinRL's 35 %. |
| DeepSeek retry logic | Flash model rate-limit retries handled gracefully in the adapter. |

### Comparison layer

`compare.py` reads from `.state/decisions/` so it works while the batch is still running —
partial reports are generated automatically without waiting for 100 % completion.

---

## 3. What Didn't Work / Surprises

### 3.1 FinRL date-alignment bug (`ValueError: could not broadcast input array`)

**What happened:** The optimal-scenario run (10 stocks) mixed tickers with different
history start dates (AAPL/MSFT/NVDA from 2018, TSLA/AMD/META/NFLX/INTC from 2017).
`FeatureEngineer` computes `close_60_sma` on the combined DataFrame and then drops rows
with NaN — but the SMA requires 60 rows of history, so the combined dataset effectively
starts at the *latest* earliest date across all tickers. When history starts differ, some
tickers are silently dropped, producing a mismatched state shape.

**Fix:** Re-download all tickers from the same start date (2017-01-01).
**Rule going forward:** All tickers in a pool must share the same `data_start` date before
running `FeatureEngineer`. Verify with `df.groupby('tic').size()` — min should equal max.

### 3.2 TradingAgents cost accounting

`cost_usd` and `prompt_tokens` stored in each `.state/decisions/*.json` file are
**cumulative within the current run**, not per-call. The first decision of each run holds
the per-call cost; subsequent decisions hold the running total. Summing across all files
gives a wildly inflated number (~$1 282 instead of ~$10).

**Fix:** Use `progress.json`'s final `cost_usd` per run and add run totals manually.
Track per-call cost as `first_decision_of_run.cost_usd ≈ avg_call_cost`.

### 3.3 TradingAgents Mon-Fri vs NYSE calendar (325 vs 305 pairs)

The original batch generated pairs from a Mon–Fri calendar that included holidays like
2024-01-01 and 2024-02-19. The LLM happily runs on holidays (it has no concept of market
closure). The actual NYSE trading day count for Q1-2024 is 61 days × 5 tickers = 305 pairs.

**Fix:** Use `AAPL.csv` dates as the ground-truth NYSE calendar — no external dependency
and no ambiguity.

### 3.4 Git nested `.git` in vendor directory

`arena/adapters/tradingagents/vendor/TradingAgents/` is a git-cloned repo and contains
its own `.git`. Trying to `git add arena/` silently treated it as a submodule, excluding
all vendor files and breaking the adapter at runtime on a fresh clone.

**Fix:** Add `adapters/tradingagents/vendor/` to `arena/.gitignore`. The package is
installed via `pip install -e ./arena/adapters/tradingagents/vendor/TradingAgents` at
environment-setup time, not vendored into the repo.

### 3.5 FinRL weight matrix is share-count-weighted, not market-value-weighted

`daily_weight_matrix` in the result JSON divides share counts by total shares held,
not shares × price / portfolio_value. In a single-dominated portfolio this is a close
approximation, but in a balanced portfolio it will materially misstate weights.

**Fix for next batch:** Compute weights as
`shares_i × close_price_i / sum(shares_j × close_price_j + cash)`.

### 3.6 TradingAgents process silently dies (~41 h run)

The batch died twice with no error — the OS killed the long-running process. `nohup`
alone is insufficient on the cluster; the process was still evictable under memory pressure.

**Fix for next batch:** Use `screen` or a `tmux` session with `remain-on-exit on`, or
submit via the cluster's job scheduler (SLURM/PBS) with a wall-time well above the
expected runtime. Add a heartbeat file (touch `/tmp/ta_alive`) every 10 decisions so
monitoring scripts can detect stale processes faster than checking `ps aux`.

---

## 4. Key Numbers

| Metric | Value |
|---|---|
| FinRL unconstrained return | −6.97 % |
| FinRL constrained return (30 %/5 %) | +0.81 % |
| FinRL optimal scenario (10 stocks, 2021-H1) | +19.94 %, Sharpe +1.55 |
| Equal-weight buy-and-hold (2024-Q1) | +24.7 % |
| TradingAgents decisions (315 NYSE + 5 holiday) | 320 total |
| TradingAgents directional accuracy | 72 % (3/5 tickers clearly right, 2 neutral) |
| TradingAgents FinRL agreement rate | 43 % (130 / 300 overlapping pairs) |
| TradingAgents avg latency | 7.9 min / decision |
| TradingAgents avg cost | $4.07 / decision |
| TradingAgents total cost | $10.19 |
| FinRL training time | 139 s (CPU, 50 k timesteps) |
| FinRL inference time | < 1 ms / decision |

---

## 5. Onboarding Checklist

A new contributor should be able to reproduce all results with the following steps.

### Environment setup

```bash
# FinRL environment
conda create -n finrl_real python=3.10 -y
conda activate finrl_real
pip install finrl stable-baselines3 torch pandas matplotlib

# TradingAgents environment (Python ≥ 3.12 required by the upstream package)
conda create -n tradingagents_real python=3.12 -y
conda activate tradingagents_real
pip install -e ./arena/adapters/tradingagents/vendor/TradingAgents
pip install langchain-openai langchain-community openai python-dotenv
```

### API keys

Create `.claude/settings.local.json` (gitignored):

```json
{
  "env": {
    "DEEPSEEK_API_KEY": "<your key>",
    "GITHUB_TOKEN": "<your token>"
  }
}
```

Or export directly:
```bash
export DEEPSEEK_API_KEY=sk-...
```

### Run FinRL (5 min)

```bash
conda run -n finrl_real python arena/adapters/finrl/run.py \
  --task portfolio_2024q1 --timesteps 50000

# With position constraints:
conda run -n finrl_real python arena/adapters/finrl/run.py \
  --task portfolio_2024q1 --timesteps 50000 --constrained

# Result: arena/results/portfolio_2024q1/finrl.json
```

### Run TradingAgents (warn: ~41 h, ~$10)

```bash
# Use screen or tmux — this will be killed if run naked in SSH
screen -S ta_run
conda run -n tradingagents_real --no-capture-output \
  python arena/adapters/tradingagents/run.py
# Ctrl-A D to detach; screen -r ta_run to reattach

# If the process dies, just re-run — it resumes from checkpoint automatically.
# Result: arena/results/portfolio_2024q1/tradingagents.json
```

### Generate reports and figures

```bash
# Text reports (markdown)
conda run -n finrl_real python arena/compare.py \
  --task portfolio_2024q1 --out-dir arena/reports/2024q1

# Figures (7 PNG files)
conda run -n finrl_real python arena/scripts/plot_reports.py
# or run the scratchpad script directly — see arena/reports/2024q1/ for outputs
```

### Verify results

```bash
python3 -c "
import json; from pathlib import Path
r = json.loads(Path('arena/results/portfolio_2024q1/finrl.json').read_text())
print('Return:', r['extracted']['total_return'])   # expect ~-0.0697
r = json.loads(Path('arena/results/portfolio_2024q1/tradingagents.json').read_text())
print('Decisions:', len(r['native_output']['decisions']))  # expect ~320
"
```

---

## 6. Next-Batch Candidates

Ordered by estimated impact / effort ratio.

### 6.1 Fix FinRL weight computation (high impact, low effort)

Current `daily_weight_matrix` uses share-count weights, not market-value weights.
One 10-line fix in `run.py` (multiply `shares` by `prices`); re-run takes 5 min.
Will change all downstream figures and the constrained/unconstrained comparison slightly.

### 6.2 Multi-seed FinRL training (high impact, medium effort)

The current result is a single seed (seed=0). PPO policy gradient is high-variance;
results can change significantly between seeds. Run 5 seeds, report mean ± std.
Estimated: 5 × 3 min = 15 min compute, plus a small `seeds.py` wrapper.

### 6.3 Extend to a larger stock universe (medium impact, medium effort)

Current: 5 stocks. Suggested next batches:
- **10 stocks, 2024-Q1:** Add TSLA, AMD, META, NFLX, INTC to the existing task.
  Stress-tests both FinRL's portfolio optimisation and TradingAgents' coverage breadth.
  TradingAgents cost: ~2× current = ~$20.
- **S&P 100 subset (20–30 stocks):** Tests whether FinRL over-concentrates when given
  more options, and whether TradingAgents signal quality degrades on less-covered stocks.

### 6.4 TradingAgents cost reduction (medium impact, medium effort)

At $4 / decision, a 30-stock × 252-day annual run costs ~$30 000. Three levers:
- **Summarise analyst reports before debate:** Instead of passing full ~8 000-token
  market/sentiment/news/fundamentals reports to the debate stage, pass 200-token
  summaries. Estimated 60 % token reduction.
- **Reduce from 18 to 10 LLM calls:** Skip the second bull/bear round if the first
  round reaches consensus (confidence > 0.8).
- **Cache identical market states:** On low-volatility days, many tickers see near-
  identical macro context. A fuzzy cache on the market-report embedding could reuse
  existing decisions.

### 6.5 Complementarity ensemble (medium impact, high effort)

The radar chart (Report 3, Fig 3) suggests the natural next step: an ensemble that uses
**FinRL for baseline allocation** and **TradingAgents to override** on high-conviction
divergence days (e.g., TA calls SELL while FinRL allocation is >50 %).

Prototype rule:
```
if TA_signal == SELL and finrl_weight > 0.40:
    override_weight = min(finrl_weight, 0.15)   # force reduce
elif TA_signal == BUY and finrl_weight < 0.05:
    override_weight = 0.10                       # force include
else:
    override_weight = finrl_weight               # follow RL
```
Backtest this rule on the 2024-Q1 data we already have (no new API spend required).

### 6.6 More market regimes (medium impact, high effort)

The 2024-Q1 test period is a single regime (mild correction after 2023 bull).
Add at least two more:
- **2022-Q1 (bear market):** NASDAQ −20 %, crypto crash. Tests TradingAgents in strongly
  trending-down markets and FinRL's behaviour when its training distribution (2019–21)
  diverges maximally from the test.
- **2020-Q1 (COVID crash + recovery):** Maximum volatility; tests whether TradingAgents'
  news-reading advantage is largest during high-news-flow periods.

### 6.7 FinMem or memory-augmented LLM adapter (low priority, high effort)

TradingAgents has no cross-day memory — each decision is made from scratch.
A FinMem-style agent that accumulates a rolling summary of its own past decisions and
their outcomes could reduce flip-flopping (e.g., alternating BUY/SELL/BUY on NVDA) and
potentially improve directional accuracy. Listed in the community task list in README.

---

## 7. Architecture Decisions Made

These decisions were made during the pilot and should be treated as settled unless a
specific experiment contradicts them.

| Decision | Rationale |
|---|---|
| Two separate conda envs (`finrl_real`, `tradingagents_real`) | TradingAgents requires Python ≥ 3.12; FinRL works best on 3.10. Sharing an env causes langchain/gym version conflicts. |
| Per-call `.state/decisions/*.json` cache (not just `progress.json`) | Allows partial-batch reporting via `compare.py` without waiting for 100 % completion. Also survives process crashes with zero re-work. |
| `ConstrainedStockTradingEnv` subclassing `step()`, not modifying training | Position constraints are a deployment-time guardrail, not a training objective. Keeping them out of the reward function avoids reward shaping entanglement. |
| DeepSeek-flash for all TradingAgents roles | Budget-first: flash is 10× cheaper than DeepSeek-V3. Direction accuracy (72 %) is already strong. Upgrade to V3 only if accuracy on a specific hard ticker is demonstrably insufficient. |
| Equal-weight BnH as the primary benchmark | Zero parameters, zero look-ahead, implementable by anyone. Outperforming it is the minimum bar for any agent to claim "added value". |

---

## 8. Open Questions

1. **Is the TA direction accuracy (72 %) statistically significant?** With 5 tickers and
   one quarter, the sample is too small for a binomial test with useful power. Need ≥ 3
   quarters × 10 tickers before making a claim.

2. **Does FinRL AAPL over-concentration persist across seeds?** If all 5 seeds converge
   to >80 % AAPL, the bias is structural (training data); if it varies, a lucky seed might
   diversify naturally.

3. **What is the correct comparison unit?** FinRL operates on the whole portfolio; TradingAgents
   operates on individual stocks. The "43 % agreement" number treats FinRL's weight-change
   direction as a per-stock signal, which is a lossy projection. A fairer comparison might
   compare portfolio-level P&L with a TradingAgents-driven allocation (equal-weight among
   BUY signals, zero weight on SELL signals).

4. **Is the GOOGL latency (9.6 min) a token-count issue or a model-saturation issue?**
   Log the token count per analyst call for GOOGL vs AAPL to diagnose.
