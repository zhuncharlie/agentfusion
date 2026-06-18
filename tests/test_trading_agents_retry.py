"""Regression test for the empty/unparseable-DeepSeek-response crash (see CHANGELOG).

A live run crashed mid-backtest because one DeepSeek call returned an empty content
string and the original code had no retry/fallback path. These tests mock
`requests.post` so they run with no network access and no API cost.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from agentfusion.agents.trading_agents import TradingAgentsAgent


def _mock_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


def _agent(tmp_path, monkeypatch, name: str) -> TradingAgentsAgent:
    monkeypatch.setattr(
        "agentfusion.agents.trading_agents.COST_LOG_PATH", tmp_path / f"{name}_cost.json"
    )
    return TradingAgentsAgent(ticker="TEST", cache_path=tmp_path / f"{name}_signals.csv", api_key="dummy")


def _history() -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0 + i for i in range(25)]})


def test_retries_then_succeeds_on_a_later_attempt(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, "ok")
    responses = [
        _mock_response(""),  # empty content -- the failure mode that crashed the live run
        _mock_response("not json either"),
        _mock_response('{"action": "BUY", "confidence": 0.8, "rationale": "ok"}'),
    ]
    with patch("agentfusion.agents.trading_agents.requests.post", side_effect=responses):
        result = agent._call_deepseek("2023-01-03", _history())

    assert result == {"action": "BUY", "confidence": 0.8, "rationale": "ok"}
    cost_log = json.loads((tmp_path / "ok_cost.json").read_text())
    assert cost_log["calls"] == 3  # all 3 attempts are billed -- DeepSeek charged for each


def test_falls_back_to_hold_after_exhausting_all_retries(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, "fail")
    with patch("agentfusion.agents.trading_agents.requests.post", return_value=_mock_response("")):
        result = agent._call_deepseek("2023-01-03", _history())

    assert result["action"] == "HOLD"
    assert result["confidence"] == 0.0
    assert "parse_failed_after_retries" in result["rationale"]


def test_never_crashes_the_caller_even_if_every_attempt_is_garbage(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, "garbage")
    with patch("agentfusion.agents.trading_agents.requests.post", return_value=_mock_response("<<not json>>")):
        signal = agent.decide({"date": pd.Timestamp("2023-01-03"), "history": _history()})

    assert signal.action.value == "HOLD"
    assert signal.confidence == 0.0
