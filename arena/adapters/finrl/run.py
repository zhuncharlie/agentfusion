"""Real FinRL portfolio adapter for the Arena pilot.

Run inside the `finrl_real` conda env (see README.md for setup):
    conda run -n finrl_real python arena/adapters/finrl/run.py [--task TASK] [--timesteps N]

Tasks:
    portfolio_2024q1  (default) — 5 stocks × 60 trading days, 2024-Q1
    portfolio_2023h1  — 3 stocks × 2023-H1 (original pilot, kept for reference)

FinRL is fundamentally a multi-asset portfolio system (one continuous action per
ticker, trained jointly), so this adapter does NOT force it into a single-asset
BUY/HOLD/SELL contract. It calls FinRL's real preprocessing/env/training API end
to end and writes the result into the Arena schema (arena/SCHEMA.md), preserving
FinRL's native output unedited.
"""
import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

TASK_CONFIGS = {
    "portfolio_2024q1": {
        "tickers":     ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        "train_start": "2020-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2024-03-31",
    },
    "portfolio_2023h1": {
        "tickers":     ["AAPL", "MSFT", "NVDA"],
        "train_start": "2020-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2023-01-01",
        "test_end":    "2023-06-30",
    },
}


def _stub_finrl_package():
    """finrl/__init__.py unconditionally imports finrl.test/trade/train, which pull in
    Alpaca/WRDS/exchange_calendars live-trading deps we don't need and can't cleanly
    install here. None of the 3 submodules we actually use depend on that chain
    (verified directly against installed package source) -- so we stub the top-level
    package to skip running its __init__.py, while still letting submodule imports
    resolve correctly via a real __path__.
    """
    spec = importlib.util.find_spec("finrl")
    stub = types.ModuleType("finrl")
    stub.__path__ = list(spec.submodule_search_locations)
    sys.modules["finrl"] = stub


_stub_finrl_package()

from finrl import config  # noqa: E402
from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402
from finrl.meta.preprocessor.preprocessors import FeatureEngineer  # noqa: E402


class ConstrainedStockTradingEnv(StockTradingEnv):
    """StockTradingEnv with hard position-size constraints enforced at every step.

    Constraints (applied before the parent processes buy actions):
      - MAX_SINGLE_WEIGHT: no single stock may exceed this fraction of portfolio value
      - MIN_CASH_RATIO:    cash may not fall below this fraction of portfolio value

    Implementation: override step() to pre-clip the raw PPO actions (which are in
    [-1, 1] space, later scaled by hmax) so that the resulting share purchases cannot
    violate either constraint.  Sell actions are left untouched (selling always reduces
    concentration; we do not enforce minimum stock weights).
    """
    MAX_SINGLE_WEIGHT: float = 0.30   # max 30% in any single stock
    MIN_CASH_RATIO:    float = 0.05   # min 5% cash at all times

    def step(self, actions):
        # terminal check mirrors the parent: if already at final day, skip constraint
        # logic and let the parent handle the terminal reporting path.
        if self.day < len(self.df.index.unique()) - 1:
            actions = self._clip_to_constraints(np.array(actions, dtype=float))
        return super().step(actions)

    def _clip_to_constraints(self, actions: np.ndarray) -> np.ndarray:
        """Pre-clip raw PPO actions ([-1,1] space) to satisfy weight constraints.

        The parent will multiply by hmax and convert to int, so we work in
        share-delta space internally, then divide back by hmax before returning.
        """
        cash   = float(self.state[0])
        prices = np.array(self.state[1:1 + self.stock_dim], dtype=float)
        shares = np.array(self.state[1 + self.stock_dim:1 + 2 * self.stock_dim],
                          dtype=float)
        V = cash + float(np.dot(shares, prices))

        if V <= 0 or np.any(prices <= 0):
            return actions

        # Convert raw actions to share-delta space for constraint arithmetic
        acts = actions * self.hmax   # float deltas

        # ── Per-stock weight cap ────────────────────────────────────────────
        for i in range(self.stock_dim):
            if acts[i] > 0:
                # max additional shares before hitting the weight cap
                max_allowed = max(0.0, self.MAX_SINGLE_WEIGHT * V / prices[i] - shares[i])
                acts[i] = min(acts[i], max_allowed)

        # ── Cash floor: total buy spend ≤ (cash - MIN_CASH_RATIO * V) ──────
        spendable = max(0.0, cash - self.MIN_CASH_RATIO * V)
        total_buy_cost = sum(
            acts[i] * prices[i] * (1.0 + self.buy_cost_pct[i])
            for i in range(self.stock_dim) if acts[i] > 0
        )
        if total_buy_cost > spendable and total_buy_cost > 0:
            scale = spendable / total_buy_cost
            for i in range(self.stock_dim):
                if acts[i] > 0:
                    acts[i] *= scale

        # Return un-scaled so the parent's `actions * hmax` restores share deltas
        return acts / self.hmax


def load_long_format(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for tic in tickers:
        df = pd.read_csv(ROOT / "data" / "raw" / f"{tic}.csv", parse_dates=["date"])
        df["tic"] = tic
        frames.append(df[["date", "tic", "open", "high", "low", "close", "volume"]])
    long_df = pd.concat(frames, ignore_index=True)
    return long_df.sort_values(["date", "tic"]).reset_index(drop=True)


def _reindex_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """StockTradingEnv indexes rows by an integer trading-day counter (same value for
    all tickers on a given date), not by date string -- this is what `df.index.unique()`
    and the multi-ticker slicing logic inside the env expect."""
    df = df.copy()
    df.index = df["date"].astype(str).factorize()[0]
    return df


def build_env_kwargs(stock_dim: int) -> dict:
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim
    return dict(
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=100_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=config.INDICATORS,
    )


def compute_metrics(account_memory: pd.DataFrame) -> dict:
    daily_returns = account_memory["account_value"].pct_change().dropna()
    total_return  = account_memory["account_value"].iloc[-1] / 100_000 - 1
    sharpe        = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0 else None
    )
    running_max = account_memory["account_value"].cummax()
    mdd         = ((account_memory["account_value"] / running_max) - 1).min()

    calmar = None
    if mdd < 0 and sharpe is not None:
        annual_return = (1 + total_return) ** (252 / max(len(daily_returns), 1)) - 1
        calmar = annual_return / abs(mdd)

    # Daily turnover: sum of absolute weight changes ÷ 2 per day (approximated from
    # account_value changes; exact turnover requires portfolio weights not available here)
    return {
        "total_return": float(total_return),
        "sharpe":       float(sharpe) if sharpe is not None else None,
        "mdd":          float(mdd),
        "calmar":       float(calmar) if calmar is not None else None,
    }


def compute_daily_weights(actions_memory: pd.DataFrame,
                          account_memory: pd.DataFrame,
                          processed_df: pd.DataFrame,
                          tickers: list[str],
                          test_start: str,
                          test_end: str) -> list[dict]:
    """Build a daily portfolio weight matrix with correct market-value weights.

    actions_memory stores buy/sell DELTAS per step (not cumulative holdings).
    We reconstruct positions by cumulative summing and price from processed_df.
    """
    action_records = actions_memory.reset_index()
    action_records.columns = ["date"] + list(actions_memory.columns)
    action_records["date"] = action_records["date"].astype(str)

    acct_by_date = {row["date"]: row["account_value"]
                    for row in account_memory.to_dict(orient="records")}

    # Close prices for the test period, keyed by (date, tic)
    test_df = processed_df[(processed_df["date"] >= test_start) & (processed_df["date"] <= test_end)]
    close_prices: dict[str, dict[str, float]] = {}
    for tic in tickers:
        close_prices[tic] = {
            row["date"]: row["close"]
            for _, row in test_df[test_df["tic"] == tic][["date", "close"]].iterrows()
        }

    holdings: dict[str, float] = {t: 0.0 for t in tickers}
    weight_rows = []

    for _, row in action_records.iterrows():
        date = row["date"]
        for t in tickers:
            holdings[t] += float(row.get(t, 0))

        portfolio_val = acct_by_date.get(date, 100_000)
        equity = {t: holdings[t] * close_prices[t].get(date, 0.0) for t in tickers}
        total_equity = sum(equity.values())
        cash_approx  = max(0.0, portfolio_val - total_equity)

        weights = {t: round(equity[t] / portfolio_val, 4) if portfolio_val > 0 else 0.0
                   for t in tickers}
        weights["cash"] = round(cash_approx / portfolio_val, 4) if portfolio_val > 0 else 0.0
        weight_rows.append({"date": date, **weights})

    return weight_rows


def _print_comparison(label_a: str, res_a: dict, label_b: str, res_b: dict) -> None:
    """Print a side-by-side metric comparison table."""
    def _m(r, key, fmt=".3f"):
        v = r.get("extracted", {}).get(key)
        return format(v, fmt) if v is not None else "—"

    def _max_weight(r):
        dw = r.get("native_output", {}).get("daily_weight_matrix", [])
        if not dw:
            return "—"
        tickers = [k for k in dw[0] if k not in ("date", "cash")]
        mx = max(max(row.get(t, 0) for t in tickers) for row in dw)
        return f"{mx:.1%}"

    def _avg_aapl(r):
        dw = r.get("native_output", {}).get("daily_weight_matrix", [])
        if not dw:
            return "—"
        avg = sum(row.get("AAPL", 0) for row in dw) / len(dw)
        return f"{avg:.1%}"

    rows = [
        ("Total Return",        _m(res_a, "total_return", ".3%"),    _m(res_b, "total_return", ".3%")),
        ("Sharpe (ann.)",       _m(res_a, "sharpe"),                  _m(res_b, "sharpe")),
        ("Max Drawdown",        _m(res_a, "mdd", ".3%"),              _m(res_b, "mdd", ".3%")),
        ("Calmar Ratio",        _m(res_a, "calmar"),                  _m(res_b, "calmar")),
        ("Max single-stock wt", _max_weight(res_a),                   _max_weight(res_b)),
        ("Avg AAPL weight",     _avg_aapl(res_a),                     _avg_aapl(res_b)),
    ]
    col = 22
    print(f"\n{'─'*70}")
    print(f"  Before/After Constraints Comparison")
    print(f"{'─'*70}")
    print(f"  {'Metric':<24} {label_a:>{col}} {label_b:>{col}}")
    print(f"  {'-'*24} {'-'*col} {'-'*col}")
    for name, va, vb in rows:
        print(f"  {name:<24} {va:>{col}} {vb:>{col}}")
    print(f"{'─'*70}\n")


def main(task_id: str, total_timesteps: int, constrained: bool = False) -> None:
    cfg = TASK_CONFIGS[task_id]
    tickers    = cfg["tickers"]
    train_start, train_end = cfg["train_start"], cfg["train_end"]
    test_start,  test_end  = cfg["test_start"],  cfg["test_end"]

    env_class  = ConstrainedStockTradingEnv if constrained else StockTradingEnv
    label      = "constrained" if constrained else "unconstrained"
    out_file   = "finrl_constrained.json" if constrained else "finrl.json"

    if constrained:
        print(
            f"[ConstrainedStockTradingEnv] "
            f"max_single_weight={ConstrainedStockTradingEnv.MAX_SINGLE_WEIGHT:.0%}  "
            f"min_cash={ConstrainedStockTradingEnv.MIN_CASH_RATIO:.0%}",
            flush=True,
        )

    t0      = time.time()
    long_df = load_long_format(tickers)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
    )
    processed       = fe.preprocess_data(long_df)
    processed["date"] = processed["date"].astype(str)
    processed       = processed.sort_values(["date", "tic"]).reset_index(drop=True)

    train_df = processed[(processed["date"] >= train_start) & (processed["date"] <= train_end)]
    test_df  = processed[(processed["date"] >= test_start)  & (processed["date"] <= test_end)]
    train_df = _reindex_by_day(train_df.reset_index(drop=True))
    test_df  = _reindex_by_day(test_df.reset_index(drop=True))

    env_kwargs  = build_env_kwargs(stock_dim=len(tickers))
    train_env   = env_class(df=train_df, **env_kwargs)
    env_train, _ = train_env.get_sb_env()

    agent       = DRLAgent(env=env_train)
    model       = agent.get_model("ppo", seed=0, verbose=0)
    print(f"Training PPO ({label}) for {total_timesteps} timesteps on "
          f"{tickers} ({train_start}..{train_end})...", flush=True)
    trained_model = DRLAgent.train_model(model, tb_log_name="ppo_arena", total_timesteps=total_timesteps)

    test_env  = env_class(df=test_df, **env_kwargs)
    account_memory, actions_memory = DRLAgent.DRL_prediction(trained_model, test_env, deterministic=True)

    elapsed = time.time() - t0

    account_memory["date"] = account_memory["date"].astype(str)
    metrics  = compute_metrics(account_memory)
    daily_weights = compute_daily_weights(
        actions_memory, account_memory, processed, tickers, test_start, test_end
    )
    actions_records = json.loads(
        json.dumps(actions_memory.reset_index().to_dict(orient="records"), default=str)
    )

    constraint_note = ""
    if constrained:
        constraint_note = (
            f" Constraints active: max {ConstrainedStockTradingEnv.MAX_SINGLE_WEIGHT:.0%} "
            f"per stock, min {ConstrainedStockTradingEnv.MIN_CASH_RATIO:.0%} cash "
            f"(enforced in step() via ConstrainedStockTradingEnv)."
        )

    native_output = {
        "tickers":               tickers,
        "train_period":          [train_start, train_end],
        "test_period":           [test_start, test_end],
        "total_timesteps":       total_timesteps,
        "constrained":           constrained,
        "account_memory":        account_memory.to_dict(orient="records"),
        "actions_memory":        actions_records,
        "daily_weight_matrix":   daily_weights,
        "final_account_value":   float(account_memory["account_value"].iloc[-1]),
        "initial_account_value": 100_000.0,
        "elapsed_sec":           elapsed,
    }

    result = {
        "project":   f"FinRL{'_constrained' if constrained else ''}",
        "task_id":   task_id,
        "native_output": native_output,
        "extracted": {
            "action":           None,
            "confidence":       None,
            "predicted_return": None,
            "sharpe":           metrics["sharpe"],
            "total_return":     metrics["total_return"],
            "mdd":              metrics["mdd"],
            "calmar":           metrics.get("calmar"),
        },
        "cost_usd":      0.0,
        "latency_sec":   elapsed,
        "adapter_notes": (
            f"Real FinRL StockTradingEnv ({len(tickers)}-ticker portfolio, PPO, "
            f"{total_timesteps} timesteps, {label}).{constraint_note} "
            "Sharpe/MDD/Calmar are portfolio-level."
        ),
    }

    out_dir = ROOT / "arena" / "results" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_file
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {out_path}", flush=True)
    print(
        f"[{label}] Sharpe={metrics['sharpe']:.3f}  Return={metrics['total_return']:.3%}  "
        f"MDD={metrics['mdd']:.3%}  Calmar={metrics.get('calmar')}  "
        f"(elapsed {elapsed:.1f}s)",
        flush=True,
    )

    # If both constrained and unconstrained results exist, print comparison
    baseline_path = out_dir / "finrl.json"
    constrained_path = out_dir / "finrl_constrained.json"
    if baseline_path.exists() and constrained_path.exists():
        res_base = json.loads(baseline_path.read_text())
        res_con  = json.loads(constrained_path.read_text())
        _print_comparison("Unconstrained", res_base, "Constrained (30%/5%)", res_con)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",        default="portfolio_2024q1",
                        choices=list(TASK_CONFIGS))
    parser.add_argument("--timesteps",   type=int, default=50_000)
    parser.add_argument("--constrained", action="store_true",
                        help="Use ConstrainedStockTradingEnv (max 30%% per stock, min 5%% cash)")
    args = parser.parse_args()
    main(args.task, args.timesteps, args.constrained)
