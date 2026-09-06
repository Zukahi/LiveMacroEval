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

## Filtered vs unfiltered on the same release

`ism_mfg_2026-08_pit` runs the identical job through the point-in-time client. Both runs
passed the leakage audit; the difference is that the filtered one could not have failed it.

| | unfiltered | point-in-time |
|---|---|---|
| point estimate | 55.1 | 55.3 |
| absolute error | **0.50** | 0.70 |
| p_above_consensus | 0.45 (correct side) | 0.53 (**wrong side**) |
| arguments | 5 up / 4 down | 5 up / 5 down |
| tool calls | 13 | 10 |
| attempts to reach an unfiltered tool | n/a | 0 |

The filtered run did worse. On a single release that is worth almost nothing — the gap is
0.2 index points and the sample is one — but it is the honest direction of the result and
is recorded rather than buried.

Two things the run did show, which do not depend on the sample size:

- **Provider date filters are not sufficient.** Exa returned documents dated 2026-09-01
  despite an `endPublishedDate` of 13:59Z that day: its filter is date-granular, not
  time-granular. The local re-check caught them. Redundancy earned its keep on the first
  live call.
- **The strict undated rule has a real cost.** `get_contents` refused the S&P Global flash
  PMI press release because the page carries no publication date — and that release is the
  single most informative bearish source for this print. The agent recovered it through a
  dated Reuters write-up, but a cheaper indicator might not have a second route. Per-domain
  date resolution for known publishers would recover this without weakening the rule.

Source mix also shifts: the unfiltered agent reached newyorkfed.org and ismworld.org
directly, while the filtered agent leaned on aggregators (MarketScreener, Investinglive)
that happen to sit in Exa's index. That is a difference in what gets cited, not only in
how much.

## A series: 8 releases, no evidence of skill

`ism_clean_2026` runs eight ISM releases — Manufacturing and Services, target months
May–August 2026 — through the point-in-time client. Every one was published after the
model's training cutoff, so recall of the answer is unavailable and the date filter is
the only thing that has to hold. All eight passed the leakage audit.

```bash
python run_evidence_series.py --series ism_clean_2026 --skip-existing
python run_evidence_series.py --series ism_clean_2026 --score-only
```

| job | consensus | actual | model | model err | consensus err | side |
|---|---|---|---|---|---|---|
| mfg 2026-05 | 53.0 | 54.0 | 53.5 | 0.5 | 1.0 | right |
| mfg 2026-06 | 53.8 | 53.3 | 53.7 | 0.4 | 0.5 | right |
| mfg 2026-07 | 54.0 | 55.6 | 55.0 | 0.6 | 1.6 | right |
| mfg 2026-08 | 55.2 | 54.6 | 55.3 | 0.7 | 0.6 | wrong |
| svc 2026-05 | 53.7 | 54.5 | 53.4 | 1.1 | 0.8 | wrong |
| svc 2026-06 | 54.0 | 54.0 | 54.2 | 0.2 | 0.0 | tie |
| svc 2026-07 | 54.5 | 54.1 | 54.6 | 0.5 | 0.4 | wrong |
| svc 2026-08 | 54.3 | 55.4 | 54.5 | 0.9 | 1.1 | right |

Mean absolute error: **model 0.613, consensus 0.750**. Correct side of consensus on 4 of
7 (the June services print landed exactly on consensus, where no side is correct).

That headline flatters the model, and the detail withdraws the flattery:

- The model beat consensus on **4 of 8** releases — a coin flip. Sign test p = 1.00.
- The median difference in absolute error is **0.000**. The mean advantage of 0.137 is
  carried by a single release: July manufacturing, where consensus missed by 1.6 and the
  model by 0.6. **Drop that one release and the gap is 0.614 against 0.629** — nothing.
- The split by indicator runs in opposite directions: manufacturing 3 of 4 with MAE 0.550
  against 0.925, services 1 of 4 with MAE 0.675 against 0.575. With four observations each,
  that is as likely to be noise as a real difference between the two surveys.

**The honest reading is that eight releases show no measurable edge over consensus.** What
they do show is that the pipeline produces clean, sourced, auditable forecasts at a cost of
roughly two minutes and a dozen searches per release — which is the precondition for a
measurement, not the measurement itself. Distinguishing a 0.1-point edge from noise at this
error level needs on the order of a hundred releases, which means widening to the other 22
variables in `variables.py` and running forward as releases arrive, rather than backfilling
a benchmark that ends where the training data begins.

One operational note from the run: a query like "ISM services forecast preview August 2026"
returned 8 results and all 8 were blocked, because previews cluster in the days just before
a release. The filter is behaving correctly, but it means the agent is systematically
poorer in exactly the sources a human forecaster would reach for first.
