"""
Manual runner for evidence mode: one job (= one indicator), one model, one run.

Usage (from LiveMacro/backend):

    python run_evidence_once.py --job ism_2026-04_backtest
    python run_evidence_once.py --job ism_2026-04_backtest --model claude-code-agent
    python run_evidence_once.py --list
    python run_evidence_once.py --job X --dry-run     # print the prompt, call nothing

Jobs live in config/jobs_evidence.json. Results are appended to
data_evidence/model_<name>/. Nothing here touches the original nowcast mode.
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG_DIR, LOCAL_TZ, get_logger
from forecasting_evidence import _build_prompt, forecast_evidence_once, get_indicator
from storage_evidence import append_evidence

logger = get_logger(__name__)

JOBS_EVIDENCE_PATH = CONFIG_DIR / "jobs_evidence.json"

REQUIRED_JOB_FIELDS = ["id", "indicator", "target_period", "release_date", "models"]


def _load_jobs():
    if not JOBS_EVIDENCE_PATH.exists():
        raise FileNotFoundError(f"No evidence jobs file at {JOBS_EVIDENCE_PATH}")
    with JOBS_EVIDENCE_PATH.open("r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise ValueError("jobs_evidence.json must contain a list")
    return jobs


def _validate(job):
    missing = [f for f in REQUIRED_JOB_FIELDS if not job.get(f)]
    if missing:
        raise ValueError(f"job {job.get('id')!r} missing required fields: {', '.join(missing)}")
    get_indicator(job["indicator"])  # raises KeyError if the indicator is unknown


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job", help="job id from config/jobs_evidence.json")
    p.add_argument("--model", default=None, help="override the job's first model")
    p.add_argument("--list", action="store_true", help="list available jobs and exit")
    p.add_argument("--dry-run", action="store_true", help="print the prompt without calling a model")
    return p.parse_args()


def main():
    args = _parse_args()
    jobs = _load_jobs()

    if args.list:
        for job in jobs:
            print(
                f"{job.get('id'):<32} {job.get('indicator'):<26} "
                f"target={job.get('target_period')} release={job.get('release_date')} "
                f"models={','.join(job.get('models', []))}"
            )
        return 0

    if not args.job:
        print("Pass --job <id>, or --list to see what is available.", file=sys.stderr)
        return 2

    matched = [j for j in jobs if j.get("id") == args.job]
    if not matched:
        print(f"No job with id {args.job!r}. Use --list.", file=sys.stderr)
        return 2
    job = matched[0]
    _validate(job)

    model_name = args.model or job["models"][0]

    if args.dry_run:
        indicator = get_indicator(job["indicator"])
        system_msg, user_msg = _build_prompt(job, indicator)
        print("=" * 30, "SYSTEM", "=" * 30)
        print(system_msg)
        print("=" * 30, "USER", "=" * 32)
        print(user_msg)
        return 0

    now_local = datetime.now(LOCAL_TZ)
    parsed, meta = forecast_evidence_once(job, model_name, now_local)
    run_id = append_evidence(job, model_name, parsed, meta)

    consensus = job.get("consensus")
    actual = job.get("actual")
    print()
    print(f"run_id            {run_id}")
    print(f"indicator         {parsed['indicator']} ({job['target_period']})")
    print(f"point_estimate    {parsed['point_estimate']}")
    print(f"consensus         {consensus}")
    if actual is not None:
        err_model = abs(parsed["point_estimate"] - float(actual))
        line = f"actual            {actual}   (model abs err {err_model:.2f}"
        if consensus is not None:
            line += f", consensus abs err {abs(float(consensus) - float(actual)):.2f}"
        print(line + ")")
    print(f"p_above_consensus {parsed['p_above_consensus']}   confidence: {parsed['confidence']}")
    print(f"arguments         {parsed['n_args_up']} up / {parsed['n_args_down']} down")
    print(f"key_uncertainty   {parsed['key_uncertainty']}")
    leak = meta.get("leak_count", 0)
    if leak:
        print(f"LEAKAGE           {leak} cited source(s) dated on/after the cutoff — run is NOT a clean backtest")
    else:
        print("leakage           none detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
