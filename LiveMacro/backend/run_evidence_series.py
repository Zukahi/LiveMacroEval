"""
Run every job in a series and score the result against consensus.

    python run_evidence_series.py --series ism_clean_2026
    python run_evidence_series.py --series ism_clean_2026 --skip-existing
    python run_evidence_series.py --series ism_clean_2026 --score-only

One release tells you nothing about skill. The question a series answers is whether
the model beats the consensus it was shown — which is the only baseline that matters,
since consensus is free and the model is not.
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import CONFIG_DIR, LOCAL_TZ, get_logger
from forecasting_evidence import forecast_evidence_once
from storage_evidence import DATA_EVIDENCE_DIR, append_evidence

logger = get_logger(__name__)

JOBS_EVIDENCE_PATH = CONFIG_DIR / "jobs_evidence.json"


def _load_series(series):
    jobs = json.loads(JOBS_EVIDENCE_PATH.read_text(encoding="utf-8"))
    return [j for j in jobs if j.get("series") == series]


def _forecast_path(job, model_name):
    base = DATA_EVIDENCE_DIR / f"model_{model_name}"
    return base / f"{job['target_period']}_{job['indicator']}_forecast.csv"


def _existing_runs(job, model_name):
    path = _forecast_path(job, model_name)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["job_id"] == job["id"]]
    return df if not df.empty else None


def run_series(series, skip_existing=False):
    jobs = _load_series(series)
    if not jobs:
        print(f"No jobs tagged series={series!r}", file=sys.stderr)
        return 1

    print(f"Running series {series}: {len(jobs)} jobs\n")
    failures = []
    for i, job in enumerate(jobs, start=1):
        model_name = job["models"][0]
        if skip_existing and _existing_runs(job, model_name) is not None:
            print(f"[{i}/{len(jobs)}] {job['id']}: already has a run, skipping")
            continue

        print(f"[{i}/{len(jobs)}] {job['id']} ({job['indicator']} {job['target_period']}) ...", flush=True)
        try:
            parsed, meta = forecast_evidence_once(job, model_name, datetime.now(LOCAL_TZ))
            append_evidence(job, model_name, parsed, meta)
            print(
                f"    estimate {parsed['point_estimate']} | consensus {job['consensus']} "
                f"| actual {job['actual']} | p_above {parsed['p_above_consensus']} "
                f"| leaks {meta.get('leak_count', 0)}"
            )
        except Exception as e:
            logger.exception("Series job failed: %s", job["id"])
            failures.append((job["id"], repr(e)))
            print(f"    FAILED: {e}")

    if failures:
        print(f"\n{len(failures)} job(s) failed:")
        for job_id, err in failures:
            print(f"  {job_id}: {err[:160]}")
    return 0


# ---------- scoring ----------
def _collect(series):
    rows = []
    for job in _load_series(series):
        model_name = job["models"][0]
        runs = _existing_runs(job, model_name)
        if runs is None:
            continue
        # If a job was run more than once, score the latest attempt.
        latest = runs.sort_values("timestamp_local").iloc[-1]
        rows.append(
            {
                "job_id": job["id"],
                "indicator": job["indicator"],
                "target": job["target_period"],
                "consensus": float(job["consensus"]),
                "actual": float(job["actual"]),
                "model": float(latest["point_estimate"]),
                "p_above": float(latest["p_above_consensus"]),
                "confidence": latest["confidence"],
                "leak_count": int(latest.get("leak_count", 0) or 0),
                "n_args": int(latest.get("n_args_up", 0) or 0) + int(latest.get("n_args_down", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def score_series(series):
    df = _collect(series)
    if df.empty:
        print(f"No completed runs for series {series!r}.", file=sys.stderr)
        return 1

    df["model_err"] = (df["model"] - df["actual"]).abs()
    df["consensus_err"] = (df["consensus"] - df["actual"]).abs()
    df["beat_consensus"] = df["model_err"] < df["consensus_err"]
    df["surprise"] = df["actual"] - df["consensus"]
    # A release that lands exactly on consensus has no correct side.
    df["tie"] = df["surprise"] == 0
    df["direction_right"] = ~df["tie"] & (
        ((df["surprise"] > 0) & (df["p_above"] > 0.5)) | ((df["surprise"] < 0) & (df["p_above"] < 0.5))
    )

    pd.set_option("display.width", 200)
    cols = ["job_id", "consensus", "actual", "model", "model_err", "consensus_err", "beat_consensus", "p_above", "direction_right", "leak_count"]
    print(df[cols].to_string(index=False))

    clean = df[df["leak_count"] == 0]
    directional = clean[~clean["tie"]]
    n = len(clean)

    print(f"\nReleases scored: {len(df)} ({n} with a clean leakage audit)")
    if len(df) != n:
        print(f"  {len(df) - n} run(s) had leaked sources and are excluded from the aggregates below.")
    print(f"  mean absolute error, model:     {clean['model_err'].mean():.3f}")
    print(f"  mean absolute error, consensus: {clean['consensus_err'].mean():.3f}")
    print(f"  beat consensus on:              {int(clean['beat_consensus'].sum())} of {n}")
    if len(directional):
        hits = int(directional["direction_right"].sum())
        print(f"  correct side of consensus:      {hits} of {len(directional)}"
              f"{' (ties excluded)' if len(directional) != n else ''}")
    print(f"  mean |surprise| to beat:        {clean['surprise'].abs().mean():.3f}")
    print(
        "\nWith a sample this small, treat these as a description of what happened, not as a "
        "measurement of skill: a few releases cannot separate a real edge from noise."
    )
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series", required=True)
    p.add_argument("--skip-existing", action="store_true", help="do not re-run jobs that already have a run")
    p.add_argument("--score-only", action="store_true", help="score what is already stored, run nothing")
    args = p.parse_args()

    if not args.score_only:
        code = run_series(args.series, skip_existing=args.skip_existing)
        if code:
            return code
        print()
    return score_series(args.series)


if __name__ == "__main__":
    raise SystemExit(main())
