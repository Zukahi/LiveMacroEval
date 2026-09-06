# Evidence mode

A parallel forecasting path added in the `evidence` branch of this fork. The original
nowcast mode (`prompts.py` / `forecasting.py` / `storage.py`) is untouched, so the fork
stays syncable with upstream and both modes can run side by side.

## Why

Upstream asks one model call for ~24 variables and forbids explanation ("Think silently
and return ONLY the requested output format"). What lands in the CSV is bare numbers:
`raw_model_output` is a key=value line, and `citations` is empty in practice. That is the
right design for measuring nowcast accuracy at scale, and the wrong one if you want to
know *why* the model expects a given print — which is what you need to trade a release.

Evidence mode trades breadth for depth: **one job = one indicator**, and the output is
the reasoning, not just the number.

## What a run produces

The model gets the published consensus, the previous print and the release date, and
returns a single JSON object:

| field | meaning |
|---|---|
| `point_estimate` | the forecast, in the indicator's own unit |
| `p_above_consensus` | P(actual > consensus), checked for agreement with the point estimate |
| `confidence` | low / medium / high |
| `arguments_for_higher` / `arguments_for_lower` | each: claim, source name, source URL, publication date, weight in [0,1] |
| `key_uncertainty` | the single thing most likely to make the forecast wrong |
| `discarded_after_cutoff` | sources the model found but rejected as post-cutoff |

Written to `data_evidence/model_<name>/`:

- `<target>_<indicator>_forecast.csv` — one row per run
- `<target>_<indicator>_arguments.csv` — one row per cited argument, joined by `run_id`

## Leakage

For a backtest the honest question is whether the model reasoned to the answer or simply
read it. The built-in web search of GPT-5 and Claude has no hard date filter, so this is
handled by **detection, not prevention**:

- a job carries an `as_of` cutoff (normally one minute before the release);
- the prompt tells the model to treat anything published at or after it as nonexistent;
- every cited source is required to carry a real publication date, and
  `audit_leakage()` flags any dated on or after the cutoff into `leak_count` /
  `after_cutoff`.

A run with `leak_count > 0` is not a valid backtest observation — discard or re-run it.
A hard pre-cutoff search layer (Exa or Tavily, date-filtered) is the next module and is
what would turn detection into prevention.

## Usage

```bash
cd LiveMacro/backend
python run_evidence_once.py --list
python run_evidence_once.py --job ism_mfg_2026-08_backtest --dry-run   # print the prompt, call nothing
python run_evidence_once.py --job ism_mfg_2026-08_backtest
python test_evidence_parser.py                                          # 16 offline checks, no model calls
```

Jobs live in `config/jobs_evidence.json`. A job needs `id`, `indicator` (any key from
`variables.py`), `target_period`, `release_date`, `models`, and — for a backtest —
`as_of`, `consensus`, `previous` and `actual`.

## First result

`ism_mfg_2026-08_backtest` — ISM Manufacturing PMI for August 2026, released 2026-09-01,
after the model's training cutoff, so the answer cannot be recalled from training.

| | value |
|---|---|
| consensus | 55.2 |
| model point estimate | 55.1 |
| actual | 54.6 |
| model absolute error | 0.50 |
| consensus absolute error | 0.60 |
| p_above_consensus | 0.45 (correct side) |
| arguments | 5 up / 4 down, 13 tool calls |
| leakage | none — all 9 sources dated 2026-08-03 to 2026-08-31 |

One release is an anecdote, not evidence of skill. What it does establish is that the
pipeline runs end to end and that the arguments are real: the model found the genuine
tension in the data — regional Fed surveys at multi-year highs against the S&P Global
flash PMI at a five-month low — and leaned to the soft side, which is where the print
landed.
