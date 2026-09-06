"""
Evidence-mode forecasting: one indicator per run, structured JSON output.

Parallel to forecasting.py, which is left untouched. The differences that matter:
  - _parse_evidence_output() parses JSON instead of a key=value line;
  - a leakage audit flags any cited source published at or after the point-in-time
    cutoff, which is what makes backtests on already-released numbers trustworthy;
  - the result carries the arguments, not just a number.
"""

import datetime as dt
import json
import re
import time

from config import MAX_RETRY, RETRY_SLEEP_SECS, get_logger
from forecasting import _as_float, _notify_failure
from llm_clients import get_client
from prompts_evidence import (
    AGENT_INSTRUCTIONS,
    ARGUMENT_FIELDS,
    build_system_msg,
    build_user_prompt,
)
from variables import VARIABLES

logger = get_logger(__name__)

CONFIDENCE_LEVELS = {"low", "medium", "high"}


class EvidenceParseError(ValueError):
    """Raised when the model's reply is not a usable evidence object."""


# ---------- indicator lookup ----------
def get_indicator(indicator_key):
    """Find a single variable definition by key across every variable group."""
    for group_name, group_vars in VARIABLES.items():
        for var in group_vars:
            if var["key"] == indicator_key:
                return {**var, "variable_group": group_name}
    raise KeyError(f"Unknown indicator: {indicator_key}")


# ---------- parsing ----------
def _extract_json_object(raw_text):
    """
    Pull the JSON object out of a model reply. Handles a bare object, an object
    wrapped in ```json fences, and an object with prose around it.
    """
    if not raw_text or not raw_text.strip():
        raise EvidenceParseError("empty model output")

    text = raw_text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates = [fenced.group(1)]
    else:
        candidates = []

    # Widest brace span, then the last balanced object — covers prose on either side.
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise EvidenceParseError(f"no JSON object found in output: {text[:200]!r}")


def _parse_date(value):
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _normalize_argument(arg, direction, idx):
    if not isinstance(arg, dict):
        raise EvidenceParseError(f"{direction} argument #{idx} is not an object")

    missing = [f for f in ARGUMENT_FIELDS if f not in arg or arg[f] in (None, "")]
    # `direction` is implied by which list the argument came from, so fill it rather than reject.
    if "direction" in missing:
        arg = {**arg, "direction": direction}
        missing.remove("direction")
    if missing:
        raise EvidenceParseError(
            f"{direction} argument #{idx} missing fields: {', '.join(missing)}"
        )

    published = _parse_date(arg.get("published_date"))
    if published is None:
        raise EvidenceParseError(
            f"{direction} argument #{idx} has unparseable published_date="
            f"{arg.get('published_date')!r}"
        )

    weight = _as_float(arg.get("weight"))
    if weight is None or not (0.0 <= weight <= 1.0):
        raise EvidenceParseError(
            f"{direction} argument #{idx} has weight outside [0,1]: {arg.get('weight')!r}"
        )

    return {
        "claim": str(arg["claim"]).strip(),
        "direction": direction,
        "source_name": str(arg["source_name"]).strip(),
        "source_url": str(arg["source_url"]).strip(),
        "published_date": published.isoformat(),
        "weight": weight,
    }


def _parse_evidence_output(raw_text, indicator_key, min_arguments=1):
    obj = _extract_json_object(raw_text)

    point_estimate = _as_float(obj.get("point_estimate"))
    if point_estimate is None:
        raise EvidenceParseError(f"point_estimate missing or non-numeric: {obj.get('point_estimate')!r}")

    p_above = _as_float(obj.get("p_above_consensus"))
    if p_above is None or not (0.0 <= p_above <= 1.0):
        raise EvidenceParseError(
            f"p_above_consensus missing or outside [0,1]: {obj.get('p_above_consensus')!r}"
        )

    confidence = str(obj.get("confidence", "")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        raise EvidenceParseError(f"confidence must be one of {sorted(CONFIDENCE_LEVELS)}: {confidence!r}")

    returned_key = str(obj.get("indicator", "")).strip()
    if returned_key and returned_key != indicator_key:
        raise EvidenceParseError(f"indicator mismatch: asked for {indicator_key!r}, got {returned_key!r}")

    args_up = [
        _normalize_argument(a, "up", i)
        for i, a in enumerate(obj.get("arguments_for_higher") or [], start=1)
    ]
    args_down = [
        _normalize_argument(a, "down", i)
        for i, a in enumerate(obj.get("arguments_for_lower") or [], start=1)
    ]

    if len(args_up) + len(args_down) < min_arguments:
        raise EvidenceParseError(
            f"too few arguments: got {len(args_up) + len(args_down)}, need >= {min_arguments}"
        )

    discarded = obj.get("discarded_after_cutoff") or []
    if not isinstance(discarded, list):
        discarded = [str(discarded)]

    parsed = {
        "indicator": indicator_key,
        "target_month": obj.get("target_month"),
        "release_date": obj.get("release_date"),
        "point_estimate": point_estimate,
        "p_above_consensus": p_above,
        "confidence": confidence,
        "arguments": args_up + args_down,
        "n_args_up": len(args_up),
        "n_args_down": len(args_down),
        "key_uncertainty": str(obj.get("key_uncertainty", "")).strip(),
        "discarded_after_cutoff": [str(u) for u in discarded],
    }
    notes = (
        f"point={point_estimate} p_above={p_above} conf={confidence} "
        f"args_up={len(args_up)} args_down={len(args_down)}"
    )
    return parsed, notes


# ---------- consistency & leakage checks ----------
def _check_direction_consistency(parsed, consensus):
    """
    point_estimate and p_above_consensus must agree. Returns a warning string, or "".
    """
    if consensus is None:
        return ""
    point, p_above = parsed["point_estimate"], parsed["p_above_consensus"]
    if point > consensus and p_above < 0.5:
        return f"inconsistent: point {point} > consensus {consensus} but p_above={p_above}"
    if point < consensus and p_above > 0.5:
        return f"inconsistent: point {point} < consensus {consensus} but p_above={p_above}"
    return ""


def audit_leakage(parsed, cutoff_iso):
    """
    Flag cited sources published at or after the point-in-time cutoff.

    The model's own web search has no hard date filter, so this is a detection
    layer, not a prevention layer: a run with leaked_sources is not a valid
    backtest observation and should be discarded or re-run.
    """
    if not cutoff_iso:
        return {"cutoff": None, "leaked_sources": [], "leak_count": 0, "undated_count": 0}

    cutoff_date = _parse_date(cutoff_iso)
    if cutoff_date is None:
        logger.warning("Unparseable cutoff %r; skipping leakage audit", cutoff_iso)
        return {"cutoff": cutoff_iso, "leaked_sources": [], "leak_count": 0, "undated_count": 0}

    leaked, undated = [], 0
    for arg in parsed["arguments"]:
        published = _parse_date(arg["published_date"])
        if published is None:
            undated += 1
            continue
        if published >= cutoff_date:
            leaked.append(
                {
                    "source_url": arg["source_url"],
                    "published_date": arg["published_date"],
                    "claim": arg["claim"][:160],
                }
            )

    if leaked:
        logger.error(
            "LEAKAGE: %d of %d cited sources are dated on/after cutoff %s",
            len(leaked), len(parsed["arguments"]), cutoff_date.isoformat(),
        )
        for item in leaked:
            logger.error("  leaked: %s (%s)", item["source_url"], item["published_date"])

    return {
        "cutoff": cutoff_iso,
        "leaked_sources": leaked,
        "leak_count": len(leaked),
        "undated_count": undated,
    }


# ---------- prompt assembly ----------
def _build_prompt(job, indicator):
    as_of = job.get("as_of")
    system_msg = build_system_msg(as_of_iso=as_of)
    user_msg = build_user_prompt(
        indicator=indicator,
        target_month_iso=job["target_period"],
        release_date_iso=job["release_date"],
        consensus=job.get("consensus"),
        previous=job.get("previous"),
        as_of_iso=as_of,
        min_arguments=int(job.get("min_arguments", 3)),
    )
    return system_msg, user_msg


# ---------- main entry point ----------
def forecast_evidence_once(job, model_name, now_local):
    """
    Run one evidence forecast for 1 job (= 1 indicator) + 1 model.
    Returns (parsed, meta).
    """
    indicator_key = job["indicator"]
    indicator = get_indicator(indicator_key)
    system_msg, user_msg = _build_prompt(job, indicator)

    client = get_client(model_name)
    client_kwargs = {}
    if model_name == "claude-code-agent":
        client_kwargs["agent_instructions"] = AGENT_INSTRUCTIONS
    elif model_name == "claude-code-agent-pit":
        # This client carries its own output contract; it needs the cutoff instead.
        if not job.get("as_of"):
            raise ValueError(
                f"job {job.get('id')!r} uses claude-code-agent-pit but has no as_of cutoff"
            )
        client_kwargs["as_of"] = job["as_of"]
        if job.get("search_provider"):
            client_kwargs["provider"] = job["search_provider"]
        client_kwargs["allow_same_day"] = bool(job.get("allow_same_day", False))

    logger.info(
        "Evidence run: job=%s model=%s indicator=%s target=%s consensus=%s as_of=%s",
        job.get("id"), model_name, indicator_key, job.get("target_period"),
        job.get("consensus"), job.get("as_of"),
    )

    last_error = None
    last_raw = None

    for attempt in range(1, MAX_RETRY + 1):
        raw = None
        try:
            logger.info("Evidence attempt %d/%d: job=%s", attempt, MAX_RETRY, job.get("id"))
            raw, citations = client(system_msg, user_msg, **client_kwargs)
            parsed, notes = _parse_evidence_output(
                raw, indicator_key, min_arguments=int(job.get("min_arguments", 3))
            )

            consistency_warning = _check_direction_consistency(parsed, _as_float(job.get("consensus")))
            if consistency_warning:
                raise EvidenceParseError(consistency_warning)

            leak = audit_leakage(parsed, job.get("as_of"))

            meta = {
                "job_id": job.get("id"),
                "model": model_name,
                "timestamp_local": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                "indicator": indicator_key,
                "variable_group": indicator["variable_group"],
                "target_month": job.get("target_period"),
                "release_date": job.get("release_date"),
                "as_of": job.get("as_of"),
                "consensus": job.get("consensus"),
                "previous": job.get("previous"),
                "actual": job.get("actual"),
                "parsed_ok": True,
                "parsed_notes": notes,
                "leak_count": leak["leak_count"],
                "undated_count": leak["undated_count"],
                "leaked_sources": leak["leaked_sources"],
                "discarded_after_cutoff": parsed["discarded_after_cutoff"],
                "raw_model_output": raw,
                "citations": "" if citations is None else str(citations),
            }
            logger.info("Evidence done: job=%s %s leak_count=%d", job.get("id"), notes, leak["leak_count"])
            return parsed, meta

        except Exception as e:
            last_error = e
            last_raw = raw
            logger.exception(
                "Evidence attempt %d/%d FAILED: job=%s error=%s", attempt, MAX_RETRY, job.get("id"), e
            )
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP_SECS)

    _notify_failure(job, model_name, last_error, last_raw)
    raise last_error
