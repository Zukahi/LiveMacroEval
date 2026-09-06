"""
CSV storage for evidence-mode runs.

Two files per (indicator, model), written with the same atomic + flock append
helpers as storage.py so the two modes can run side by side safely:

  data_evidence/model_<name>/<target>_<indicator>_forecast.csv
      one row per run: the estimate, the probability, the audit counters.

  data_evidence/model_<name>/<target>_<indicator>_arguments.csv
      one row per cited argument, joined back to the forecast by run_id.
"""

import hashlib

from config import DATA_DIR, get_logger
from storage import _append_csv_rows_locked

logger = get_logger(__name__)

DATA_EVIDENCE_DIR = DATA_DIR.parent / "data_evidence"

FORECAST_COLS = [
    "run_id",
    "timestamp_local",
    "as_of",
    "indicator",
    "variable_group",
    "target_month",
    "release_date",
    "job_id",
    "model",
    "point_estimate",
    "p_above_consensus",
    "confidence",
    "consensus",
    "previous",
    "actual",
    "n_args_up",
    "n_args_down",
    "key_uncertainty",
    "leak_count",
    "undated_sources",
    "discarded_after_cutoff",
    "parsed_ok",
    "parsed_notes",
    "raw_model_output",
]

ARGUMENT_COLS = [
    "run_id",
    "timestamp_local",
    "indicator",
    "target_month",
    "job_id",
    "model",
    "direction",
    "weight",
    "claim",
    "source_name",
    "source_url",
    "published_date",
    "after_cutoff",
]


def make_run_id(meta):
    """Stable id linking a forecast row to its argument rows."""
    seed = f"{meta['job_id']}|{meta['model']}|{meta['timestamp_local']}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _base_dir(model_name):
    base = DATA_EVIDENCE_DIR / f"model_{model_name}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _paths(job, model_name, meta):
    base = _base_dir(model_name)
    stem = f"{meta['target_month']}_{meta['indicator']}"
    return base / f"{stem}_forecast.csv", base / f"{stem}_arguments.csv"


def append_evidence(job, model_name, parsed, meta):
    forecast_path, arguments_path = _paths(job, model_name, meta)
    run_id = make_run_id(meta)

    leaked_urls = {item["source_url"] for item in meta.get("leaked_sources", [])}

    forecast_row = {
        "run_id": run_id,
        "timestamp_local": meta["timestamp_local"],
        "as_of": meta.get("as_of"),
        "indicator": meta["indicator"],
        "variable_group": meta.get("variable_group"),
        "target_month": meta["target_month"],
        "release_date": meta.get("release_date"),
        "job_id": meta["job_id"],
        "model": meta["model"],
        "point_estimate": parsed["point_estimate"],
        "p_above_consensus": parsed["p_above_consensus"],
        "confidence": parsed["confidence"],
        "consensus": meta.get("consensus"),
        "previous": meta.get("previous"),
        "actual": meta.get("actual"),
        "n_args_up": parsed["n_args_up"],
        "n_args_down": parsed["n_args_down"],
        "key_uncertainty": parsed["key_uncertainty"],
        "leak_count": meta.get("leak_count", 0),
        "undated_sources": meta.get("undated_count", 0),
        "discarded_after_cutoff": " | ".join(meta.get("discarded_after_cutoff", [])),
        "parsed_ok": meta["parsed_ok"],
        "parsed_notes": meta["parsed_notes"],
        "raw_model_output": meta["raw_model_output"],
    }
    _append_csv_rows_locked(forecast_path, FORECAST_COLS, [forecast_row])

    argument_rows = [
        {
            "run_id": run_id,
            "timestamp_local": meta["timestamp_local"],
            "indicator": meta["indicator"],
            "target_month": meta["target_month"],
            "job_id": meta["job_id"],
            "model": meta["model"],
            "direction": arg["direction"],
            "weight": arg["weight"],
            "claim": arg["claim"],
            "source_name": arg["source_name"],
            "source_url": arg["source_url"],
            "published_date": arg["published_date"],
            "after_cutoff": arg["source_url"] in leaked_urls,
        }
        for arg in parsed["arguments"]
    ]
    if argument_rows:
        _append_csv_rows_locked(arguments_path, ARGUMENT_COLS, argument_rows)

    logger.info(
        "Appended evidence run %s: 1 forecast row -> %s, %d argument rows -> %s",
        run_id, forecast_path, len(argument_rows), arguments_path,
    )
    return run_id
