# Report 1 — FinRL Portfolio Performance

**Framework:** AI4Finance-Foundation/FinRL (PPO, stable-baselines3)
**Task:** portfolio_2024q1
**Tickers:** AAPL, MSFT, NVDA, GOOGL, AMZN
**Train period:** 2020-01-01 → 2022-12-31
**Test period:** 2024-01-01 → 2024-03-31  (60 trading days)
**Training timesteps:** 50,000
**Wall-clock time:** 139.1s  |  **API cost:** $0.00 (local RL)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | -6.97% |
| Sharpe Ratio (annualized) | -1.524 |
| Max Drawdown | -13.05% |
| Calmar Ratio | -2.007 |
| Initial Capital | $100,000 |
| Final Value | $93,025.62 |

## Equity Curve (every 5th trading day)

| Date | Account Value | Daily Return |
|------|--------------|--------------|
| 2024-01-02 | $100,000.00 | — |
| 2024-01-09 | $100,413.50 | -0.151% |
| 2024-01-17 | $99,101.32 | -0.474% |
| 2024-01-24 | $105,087.66 | -0.340% |
| 2024-01-31 | $99,807.19 | -1.881% |
| 2024-02-07 | $102,335.91 | 0.054% |
| 2024-02-14 | $99,684.12 | -0.469% |
| 2024-02-22 | $99,849.88 | 1.132% |
| 2024-02-29 | $97,946.52 | -0.322% |
| 2024-03-07 | $91,686.81 | -0.052% |
| 2024-03-14 | $93,842.94 | 1.093% |
| 2024-03-21 | $92,962.75 | -4.080% |
| 2024-03-28 | $93,025.62 | -1.054% |

## Daily Portfolio Weights (every 5th trading day)

| Date | AAPL | MSFT | NVDA | GOOGL | AMZN | Cash |
|------|------|------|------|------|------|------|
| 2024-01-02 | 12.3% | 0.0% | 0.0% | 0.0% | 0.0% | 87.7% |
| 2024-01-09 | 73.3% | 0.0% | 0.0% | 0.0% | 0.0% | 26.7% |
| 2024-01-17 | 92.4% | 0.0% | 0.0% | 0.0% | 0.0% | 7.6% |
| 2024-01-24 | 96.1% | 0.8% | 0.0% | 0.0% | 0.0% | 3.2% |
| 2024-01-31 | 94.1% | 0.0% | 0.0% | 0.0% | 0.0% | 5.9% |
| 2024-02-07 | 96.1% | 0.0% | 0.0% | 0.0% | 0.0% | 3.9% |
| 2024-02-14 | 97.1% | 0.0% | 0.0% | 0.0% | 0.0% | 2.9% |
| 2024-02-22 | 98.7% | 0.0% | 0.0% | 0.0% | 1.2% | 0.1% |
| 2024-02-29 | 97.9% | 0.0% | 0.0% | 0.0% | 2.0% | 0.1% |
| 2024-03-07 | 98.9% | 0.0% | 0.0% | 0.0% | 1.0% | 0.1% |
| 2024-03-14 | 99.8% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% |
| 2024-03-21 | 99.8% | 0.0% | 0.0% | 0.2% | 0.0% | 0.0% |
| 2024-03-27 | 99.8% | 0.0% | 0.0% | 0.2% | 0.0% | 0.0% |

## Average Portfolio Weights Over Test Period

| Ticker | Avg Weight |
|--------|-----------|
| AAPL | 91.3% |
| MSFT | 0.1% |
| NVDA | 0.0% |
| GOOGL | 0.0% |
| AMZN | 0.3% |
| cash | 8.2% |

## Key Observation

> FinRL (PPO trained on 2020–2022 data) concentrated the portfolio almost entirely in AAPL (~99.8% by Q1 end), essentially ignoring MSFT, NVDA, GOOGL, and AMZN. This strategy reflects the PPO's learned prior from the training period where AAPL was a top performer. In 2024-Q1, AAPL underperformed, resulting in a -6.97% return vs a positive market period for the other four stocks.

_Adapter notes: Real FinRL StockTradingEnv (5-ticker portfolio, PPO, 50000 timesteps). Sharpe/MDD/Calmar are portfolio-level. daily_weight_matrix is approximate (share counts / total shares, not market-value weighted). For exact weights, multiply share counts by closing prices._