# Changelog

## Unreleased

### Fixed
- `TradingAgentsAgent._call_deepseek` no longer crashes the whole backtest when DeepSeek
  returns an empty or unparseable response (observed once in ~373 live calls, partway
  through the NVDA test period). It now retries up to 3 times and, if every attempt is
  still unparseable, falls back to a zero-confidence `HOLD` instead of raising. Each
  attempt is still billed and counted in the cost log. Regression coverage in
  `tests/test_trading_agents_retry.py`.
