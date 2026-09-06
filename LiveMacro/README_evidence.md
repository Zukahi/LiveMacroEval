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

### Prevention: point-in-time search

`search_pit.py` plus the `claude-code-agent-pit` client close the hole. In a PIT run the
built-in `WebSearch` and `WebFetch` are **denied**, and the agent's only window on the
world is two in-process MCP tools, `mcp__pit__search` and `mcp__pit__get_contents`, that
route through a date-filtered provider. The agent cannot read the answer because the tool
that would have to hand it over refuses.

Two redundant guards, because provider metadata is wrong often enough to matter:

1. the provider's own filter (Exa `endPublishedDate`, Tavily `end_date`);
2. a local re-check of every returned document, which drops anything dated on or after
   the cutoff — and anything undated, since an undated document cannot be proven to
   predate it.

The cutoff is bound per run by the client, not passed as a tool argument, so the agent
has no way to widen its own window. Attempts to reach a non-PIT tool are denied by a
`PreToolUse` hook and counted in the log.

Same-day documents are excluded by default. A release at 10:00 ET and a preview published
that morning often carry a date-only timestamp, so they are indistinguishable; losing a
few legitimate previews is cheaper than silently importing the answer. Set
`allow_same_day: true` on a job to relax the date rule — the clock rule still holds, so a
document timestamped after the cutoff stays blocked either way.

Setup:

```bash
export EXA_API_KEY=...      # preferred: real endPublishedDate filtering
export TAVILY_API_KEY=...   # fallback: weaker date support, local filter carries more
export PIT_SEARCH_PROVIDER=exa   # optional, to force one
```

Without a key the layer raises rather than falling back to unfiltered search — degrading
quietly would produce backtests that look clean and are not.

Run the same release both ways and compare: `ism_mfg_2026-08_backtest` (unfiltered, the
control) against `ism_mfg_2026-08_pit` (clean observation).

## Usage

```bash
cd LiveMacro/backend
python run_evidence_once.py --list
python run_evidence_once.py --job ism_mfg_2026-08_backtest --dry-run   # print the prompt, call nothing
python run_evidence_once.py --job ism_mfg_2026-08_backtest
python test_evidence_parser.py   # 16 offline checks: parser + leakage audit
python test_search_pit.py        # 14 offline checks: date filter + tool denial (HTTP stubbed)
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
