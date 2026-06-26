# Arena contracts

## Task (`arena/tasks/<task_id>.json`)

The same input every adapter answers. Adapters translate this into whatever shape their
own project expects -- there is no requirement that they consume it directly.

```json
{
  "task_id": "aapl_2023h1",
  "ticker": "AAPL",
  "history_csv": "data/raw/AAPL.csv",
  "train_start": "2020-01-01",
  "train_end": "2022-12-31",
  "start": "2023-01-01",
  "end": "2023-06-30",
  "question": "..."
}
```

## Result (`arena/results/<task_id>/<project>.json`)

Written by each adapter's `run.py`. `native_output` is intentionally unconstrained --
whatever the upstream project naturally produces (training curves, full debate
transcripts, its own backtest report) goes there unedited. `extracted` is a best-effort,
all-optional projection onto the few fields that happen to be comparable across
projects; missing ones stay `null` rather than being faked.

```json
{
  "project": "FinRL",
  "task_id": "aapl_2023h1",
  "native_output": {},
  "extracted": {
    "action": "BUY | HOLD | SELL | null",
    "confidence": "0.0-1.0 | null",
    "predicted_return": "float | null",
    "sharpe": "float | null",
    "total_return": "float | null",
    "mdd": "float | null"
  },
  "cost_usd": 0.0,
  "latency_sec": 0.0,
  "adapter_notes": "anything the adapter author wants future-self to know"
}
```

`compare.py` only ever reads this result schema -- it never imports an upstream
project's code directly, so it has no extra dependencies beyond the Arena core.
