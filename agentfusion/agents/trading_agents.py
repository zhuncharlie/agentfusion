"""TradingAgents-style multi-role LLM trading agent, implemented directly against the
DeepSeek API rather than the upstream `tradingagents` package (which requires Python
>=3.12 and a langchain stack -- the same dependency-conflict risk flagged for FinRL).

Cost control: one DeepSeek call per (ticker, date), cached to disk so repeat backtests
(ensemble, signal-agreement analysis) never re-call the API. Spend is tracked against a
hard budget cap so a bug can't silently blow through the project's API budget.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from agentfusion.base import Action, BaseAgent, Signal
from agentfusion.registry import OptimizerRegistry

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
PRICE_PER_M_INPUT = 0.14
PRICE_PER_M_OUTPUT = 0.28
BUDGET_CAP_USD = 0.60

CACHE_PATH = Path("signals/trading_agents.csv")
COST_LOG_PATH = Path("signals/_trading_agents_cost.json")

SYSTEM_PROMPT = (
    "You are a quantitative trading decision system. Internally simulate a 4-stage "
    "debate before answering: (1) Analyst summarizes the recent price/technical action; "
    "(2) Bull researcher argues for buying; (3) Bear researcher argues for caution or "
    "selling; (4) Trader weighs both arguments and makes one final call. Do not show your "
    "reasoning. Respond with ONLY a JSON object: "
    '{"action": "BUY"|"HOLD"|"SELL", "confidence": <0-1 float>, "rationale": "<=20 words"}.'
)


def _load_cost_log() -> dict:
    if COST_LOG_PATH.exists():
        return json.loads(COST_LOG_PATH.read_text())
    return {"total_usd": 0.0, "calls": 0}


def _save_cost_log(log: dict) -> None:
    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_LOG_PATH.write_text(json.dumps(log, indent=2))


def _summarize_history(history: pd.DataFrame, lookback: int = 20) -> str:
    closes = history["close"].tail(lookback).reset_index(drop=True)
    last = closes.iloc[-1]

    def pct(n):
        if len(closes) <= n:
            return None
        return (last / closes.iloc[-1 - n] - 1) * 100

    lines = [
        f"last_close={last:.2f}",
        f"chg_1d={pct(1):.2f}%" if pct(1) is not None else "chg_1d=NA",
        f"chg_5d={pct(5):.2f}%" if pct(5) is not None else "chg_5d=NA",
        f"chg_10d={pct(10):.2f}%" if pct(10) is not None else "chg_10d=NA",
        f"chg_20d={pct(20):.2f}%" if pct(20) is not None else "chg_20d=NA",
        f"volatility_10d={closes.pct_change().tail(10).std() * 100:.2f}%",
    ]
    return ", ".join(lines)


def _parse_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in DeepSeek response: {text!r}")
    obj = json.loads(match.group(0))
    action = str(obj["action"]).upper()
    if action not in ("BUY", "HOLD", "SELL"):
        raise ValueError(f"Invalid action in DeepSeek response: {action!r}")
    return {"action": action, "confidence": float(obj.get("confidence", 0.5)), "rationale": str(obj.get("rationale", ""))}


@OptimizerRegistry.register("trading_agents")
class TradingAgentsAgent(BaseAgent):
    def __init__(self, ticker: str, cache_path: Path = CACHE_PATH, api_key: str = None):
        self.ticker = ticker
        self.cache_path = Path(cache_path)
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        df = pd.read_csv(self.cache_path, parse_dates=["date"])
        df = df[df["ticker"] == self.ticker]
        return {row["date"].strftime("%Y-%m-%d"): row.to_dict() for _, row in df.iterrows()}

    def _append_cache(self, date_str: str, result: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"date": date_str, "ticker": self.ticker, **result}])
        row.to_csv(self.cache_path, mode="a", header=not self.cache_path.exists(), index=False)

    def decide(self, obs: dict) -> Signal:
        date_str = pd.Timestamp(obs["date"]).strftime("%Y-%m-%d")

        if date_str in self._cache:
            cached = self._cache[date_str]
            return Signal(Action(cached["action"]), confidence=float(cached["confidence"]))

        result = self._call_deepseek(date_str, obs["history"])
        self._cache[date_str] = {"action": result["action"], "confidence": result["confidence"], "rationale": result["rationale"]}
        self._append_cache(date_str, result)
        return Signal(Action(result["action"]), confidence=result["confidence"])

    def _call_deepseek(self, date_str: str, history: pd.DataFrame) -> dict:
        summary = _summarize_history(history)
        payload = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Ticker: {self.ticker}\nDate: {date_str}\nRecent price action: {summary}"},
            ],
        }

        last_err = None
        for attempt in range(3):
            cost_log = _load_cost_log()
            if cost_log["total_usd"] >= BUDGET_CAP_USD:
                raise RuntimeError(
                    f"DeepSeek budget cap (${BUDGET_CAP_USD}) reached after {cost_log['calls']} calls "
                    f"(${cost_log['total_usd']:.4f} spent). Aborting to protect the API budget."
                )

            resp = requests.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()

            usage = body.get("usage", {})
            call_cost = (
                usage.get("prompt_tokens", 0) * PRICE_PER_M_INPUT + usage.get("completion_tokens", 0) * PRICE_PER_M_OUTPUT
            ) / 1_000_000
            cost_log["total_usd"] += call_cost
            cost_log["calls"] += 1
            _save_cost_log(cost_log)

            try:
                return _parse_response(body["choices"][0]["message"]["content"])
            except ValueError as exc:
                last_err = exc

        # DeepSeek returned an unparseable/empty response on every retry (we've seen this
        # happen transiently). Don't let one bad day crash a 124-day backtest -- record a
        # neutral, zero-confidence HOLD so it's visibly different from a real decision.
        return {"action": "HOLD", "confidence": 0.0, "rationale": f"parse_failed_after_retries: {last_err}"}
