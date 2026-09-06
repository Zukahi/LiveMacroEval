"""
Prompts for the `evidence` mode.

Difference from prompts.py (the original nowcast mode):
  - one indicator per call instead of ~24 at once, so the agent can dig deep;
  - the model is given the published consensus, the previous print and the
    scheduled release datetime, and is asked for P(actual > consensus);
  - the output is structured JSON with explicit arguments FOR and AGAINST a
    higher-than-consensus print, each carrying a source URL and a publication
    date. The publication dates make a post-hoc leakage audit possible when the
    run is a backtest of an already-released number.

The original prompts.py is untouched, so both modes can coexist and the fork
stays syncable with upstream.
"""

import json

# Keys every argument object must carry.
ARGUMENT_FIELDS = ["claim", "direction", "source_name", "source_url", "published_date", "weight"]


def build_system_msg(as_of_iso=None):
    base = (
        "You are a web-searching macroeconomic analyst. For a SINGLE upcoming data release you "
        "gather evidence, weigh it, and produce a calibrated forecast together with the arguments "
        "that support and contradict it. "
        "Every factual claim you make must be traceable to a source you actually retrieved: give "
        "its URL and its publication date. Never invent a URL or a date. "
        "Return ONLY a single JSON object in the requested schema, with no prose around it and no "
        "markdown code fences."
    )
    if as_of_iso:
        base += (
            f"\n\nCRITICAL — POINT-IN-TIME CONSTRAINT: you are simulating an analyst standing at "
            f"{as_of_iso}. Treat anything published at or after that instant as nonexistent: do not "
            f"use it, do not cite it, and do not let it influence your estimate. This includes the "
            f"actual released value itself, any article reporting it, and any later revision. "
            f"If a search result is dated at or after that instant, discard it and say so in "
            f"`discarded_after_cutoff`. Report every source's true publication date honestly even "
            f"if that means admitting you saw a source you had to discard."
        )
    return base


def _schema_block():
    schema = {
        "indicator": "<the indicator key given above, verbatim>",
        "target_month": "YYYY-MM",
        "release_date": "YYYY-MM-DD",
        "point_estimate": "<number, in the stated unit, at most 2 decimals>",
        "p_above_consensus": "<number in [0,1]: probability the released value exceeds consensus>",
        "confidence": "<one of: low, medium, high>",
        "arguments_for_higher": [
            {
                "claim": "<one sentence, concrete and checkable>",
                "direction": "up",
                "source_name": "<publisher, e.g. 'S&P Global', 'Federal Reserve Bank of Dallas'>",
                "source_url": "<full URL you retrieved>",
                "published_date": "YYYY-MM-DD",
                "weight": "<number in [0,1]: how much this moved your estimate>",
            }
        ],
        "arguments_for_lower": [
            {
                "claim": "<one sentence, concrete and checkable>",
                "direction": "down",
                "source_name": "<publisher>",
                "source_url": "<full URL you retrieved>",
                "published_date": "YYYY-MM-DD",
                "weight": "<number in [0,1]>",
            }
        ],
        "key_uncertainty": "<the single thing most likely to make this forecast wrong>",
        "discarded_after_cutoff": [
            "<URL of any source you found but discarded because it was published at or after the cutoff; empty list if none>"
        ],
    }
    return json.dumps(schema, indent=2, ensure_ascii=False)


def _fmt(value, fallback="unknown (not provided)"):
    return fallback if value is None else str(value)


def build_user_prompt(
    indicator,
    target_month_iso,
    release_date_iso,
    consensus=None,
    previous=None,
    as_of_iso=None,
    min_arguments=3,
):
    """
    indicator: dict from variables.py — keys: key, title, unit_hint, official_sources, scale_hint
    target_month_iso: "YYYY-MM" — the month the data measures
    release_date_iso: "YYYY-MM-DD" — the official publication date
    consensus / previous: floats from the release calendar, or None
    as_of_iso: ISO timestamp the analyst is standing at. For a backtest set it to
               just before the release; for a live run leave it None.
    """
    sources_block = "\n".join(f"- {u}" for u in indicator.get("official_sources", [])) or "- (none listed)"
    scale_hint = indicator.get("scale_hint") or "(no scale hint available)"

    cutoff_block = ""
    if as_of_iso:
        cutoff_block = f"""
Point-in-time cutoff: {as_of_iso}
- This is a BACKTEST. The number has already been published in the real world, but you must forecast
  it as if standing at the cutoff.
- Any source published at or after the cutoff is off-limits. List its URL in `discarded_after_cutoff`
  rather than using it.
- Do not search for the released value, and do not "remember" it from training. If you already know
  it, that knowledge is not admissible evidence — your reasoning must stand on cited pre-cutoff sources.
"""

    return f"""Task: forecast ONE macroeconomic indicator for its upcoming release, and set out the evidence on both sides.

Indicator
- key: {indicator["key"]}
- description: {indicator["title"]}
- unit: {indicator["unit_hint"]}
- plausible scale: {scale_hint}
- target month (the month the data measures): {target_month_iso}
- scheduled release date: {release_date_iso}
- geography: United States

Market context (from the release calendar)
- consensus forecast: {_fmt(consensus)}
- previous print: {_fmt(previous)}
{cutoff_block}
Official sources for definition and calibration:
{sources_block}

Method:
1. Search for evidence that bears on this specific release: regional Fed surveys, S&P Global flash PMIs,
   sub-indices of the previous report, comparable private trackers, supply-chain and price data, sector
   news, policy and trade changes, weather or strike disruptions, and revisions to prior months.
2. Sort what you find into evidence pointing ABOVE consensus and evidence pointing BELOW it.
   Give at least {min_arguments} arguments on each side if the evidence supports that many; if one side
   genuinely has less, give fewer there rather than padding it with weak claims.
3. Weigh the two sides and produce a single point estimate plus P(actual > consensus).
   If your point estimate is above consensus, p_above_consensus must be > 0.5, and vice versa —
   the two must not contradict each other.
4. Calibration: consensus is a strong baseline. Deviate from it only as far as your evidence warrants,
   and keep p_above_consensus away from 0 and 1 unless the evidence is overwhelming.

Output format (STRICT):
- Return exactly one JSON object, nothing else. No markdown fences, no commentary before or after.
- Numbers must be bare numbers (no % sign, no thousands separators, no quotes around them).
- Every argument object must contain all of: {", ".join(ARGUMENT_FIELDS)}.
- `published_date` must be the source's real publication date in YYYY-MM-DD form. If a source shows no
  date, omit that source rather than guessing a date.

Schema:
{_schema_block()}
""".strip()


# Agent-mode instructions appended by the Claude Code Agent client. The client's
# built-in instructions demand a single key=value line, which is wrong for this
# mode, so evidence runs pass this block instead.
AGENT_INSTRUCTIONS = """AGENT INSTRUCTIONS:
You are an autonomous macro research agent working on ONE data release. Follow these steps:
1. Use only the tools allowed in this run. Use WebSearch and WebFetch to gather evidence from official
   and reputable public sources.
2. Search broadly before you conclude: at minimum look for regional Fed manufacturing/services surveys,
   flash PMIs, the sub-indices of the previous report, and any sector or policy news for the target month.
3. If a source errors or times out repeatedly, move on to another source instead of retrying it.
4. Do not use Bash, file-edit or local filesystem tools.
5. Record the real URL and real publication date of every source you rely on. If you cannot establish a
   source's publication date, do not cite it.
6. Your FINAL message must be the JSON object alone — no preamble, no explanation, no markdown fences.
"""
