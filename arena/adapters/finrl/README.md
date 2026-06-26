# FinRL adapter (real upstream package)

Wraps the actual [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
package (`pip install finrl`, version 0.3.7 at time of writing) -- not a reimplementation.
FinRL is fundamentally a **multi-asset portfolio** system (one continuous action per
ticker, trained jointly), so this adapter does not force it into AgentFusion's
single-asset `BaseAgent` interface. It runs FinRL's real `FeatureEngineer` ->
`StockTradingEnv` -> `DRLAgent` (PPO via stable-baselines3) pipeline end to end on our
actual 3-ticker universe (AAPL, MSFT, NVDA) and writes the result into the Arena
schema (`arena/SCHEMA.md`), preserving FinRL's native account/action history unedited.

## Setup

```bash
conda create -n finrl_real python=3.10 -y
conda install -n finrl_real -c conda-forge ta-lib -y   # do this BEFORE pip install finrl
conda run -n finrl_real pip install finrl gymnasium stable-baselines3 stockstats pyfolio-reloaded
conda run -n finrl_real pip install alpaca-trade-api yfinance   # see "Gotchas" below
```

## Real API used (verified against the installed 0.3.7 source, not guessed)

```python
from finrl import config                                              # config.INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=config.INDICATORS,
                      use_vix=False, use_turbulence=False)
processed = fe.preprocess_data(long_format_df)   # columns: date, tic, open, high, low, close, volume

env = StockTradingEnv(df=train_df, stock_dim=3, hmax=100, initial_amount=100_000,
                       num_stock_shares=[0,0,0], buy_cost_pct=[.001]*3, sell_cost_pct=[.001]*3,
                       reward_scaling=1e-4, state_space=1+2*3+len(config.INDICATORS)*3,
                       action_space=3, tech_indicator_list=config.INDICATORS)
env_train, _ = env.get_sb_env()                  # wraps self in a DummyVecEnv

agent = DRLAgent(env=env_train)
model = agent.get_model("ppo", seed=0)
trained = DRLAgent.train_model(model, tb_log_name="x", total_timesteps=25_000)

account_memory, actions_memory = DRLAgent.DRL_prediction(trained, test_env, deterministic=True)
# account_memory: DataFrame[date, account_value]
# actions_memory: DataFrame indexed by date, one column per ticker (share counts bought/sold)
```

`state_space = 1 + 2*stock_dim + len(tech_indicator_list)*stock_dim` (balance + close
prices per ticker + shares held per ticker + indicators per ticker) -- confirmed by
reading `StockTradingEnv._initiate_state()` directly, not from a tutorial.

## How our task maps to FinRL's inputs

Our per-ticker wide CSVs (`data/raw/{AAPL,MSFT,NVDA}.csv`, columns
`date,open,high,low,close,volume`) are concatenated into FinRL's expected long format
(one row per ticker per day, with a `tic` column) before calling `preprocess_data()`.
Train/test split matches our existing simplified FinRL agent (train 2020-01-01 to
2022-12-31, test 2023-01-01 to 2023-06-30). `StockTradingEnv` indexes rows by an
integer trading-day counter (same value across all tickers on a date), not by date
string -- `df.index = df["date"].factorize()[0]` before constructing the env.

## Gotchas

- **TA-Lib must be conda-installed before `pip install finrl`** -- pip alone usually
  fails to build TA-Lib's C extension. Confirmed by an inline comment in FinRL's own
  `requirements.txt`.
- **`import finrl` eagerly pulls in live-trading deps we never use.**
  `finrl/__init__.py` unconditionally does `from finrl.test import test; from
  finrl.trade import trade; from finrl.train import train`. `finrl.trade` imports
  `AlpacaPaperTrading`, which needs `alpaca_trade_api` and (transitively)
  `exchange_calendars`; `finrl.train` imports `finrl.meta.data_processor.DataProcessor`,
  which directly imports `processor_wrds` (needs the paid WRDS client library). None of
  this is needed for `FeatureEngineer`/`StockTradingEnv`/`DRLAgent` (verified by reading
  each file's own import list -- they only need `stockstats`, `gymnasium`,
  `stable-baselines3`, and `finrl.config`).
  - Pragmatic fix used here: `pip install alpaca-trade-api yfinance` (the two genuinely
    lightweight ones in the chain) and stub `sys.modules["finrl"]` before importing,
    so `finrl/__init__.py` never actually runs but submodule imports still resolve
    correctly via a real `__path__`:
    ```python
    import importlib.util, sys, types
    spec = importlib.util.find_spec("finrl")            # locates without executing __init__.py
    stub = types.ModuleType("finrl")
    stub.__path__ = list(spec.submodule_search_locations)
    sys.modules["finrl"] = stub
    # now `from finrl.meta... import ...` works without dragging in Alpaca/WRDS/exchange_calendars
    ```
  - This avoided needing `exchange_calendars`, `wrds`, `ccxt`, `jqdatasdk`, `ray`,
    `selenium` at all -- genuinely unused for our use case.
- **`elegantrl`/`alpaca-py`/`ccxt`/`jqdatasdk`/`wrds`/`ray`/`selenium` from the upstream
  `requirements.txt` were never installed** -- not needed by the 3 modules we actually
  call (training is via `stable-baselines3` directly, not `elegantrl`).
- No GPU needed; stable-baselines3 PPO with `MlpPolicy` runs fine on CPU (a UserWarning
  about GPU underutilization is expected and harmless if CUDA happens to be visible).

## Result

`arena/results/portfolio_2023h1/finrl.json` is the full 3-ticker portfolio result
(FinRL's native `account_memory`/`actions_memory`, 25,000 PPO timesteps, ~76s wall
clock, zero API cost since this is local RL training). Real measured result:
**Sharpe=4.115, Return=47.70%, MDD=-6.45%** over 2023-01-03 to 2023-06-30.

`arena/results/aapl_2023h1/finrl.json` is an AAPL-specific slice of the same run (the
action sequence for AAPL alone) for comparison against single-ticker adapters --
**the Sharpe/Return/MDD fields in that file are still portfolio-level**, not AAPL-only,
since FinRL doesn't decompose performance per ticker on its own. This is noted in the
file's own `adapter_notes` field.
