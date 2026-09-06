"""
Offline checks for the evidence-mode parser and leakage audit. No model calls.

    python test_evidence_parser.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecasting_evidence import (
    EvidenceParseError,
    _check_direction_consistency,
    _parse_evidence_output,
    audit_leakage,
    get_indicator,
)

KEY = "ism_manufacturing_index"


def _arg(url="https://example.org/a", date="2026-08-25", weight=0.3, direction="up"):
    return {
        "claim": "Dallas Fed manufacturing new orders rose in August.",
        "direction": direction,
        "source_name": "Federal Reserve Bank of Dallas",
        "source_url": url,
        "published_date": date,
        "weight": weight,
    }


def _obj(**over):
    base = {
        "indicator": KEY,
        "target_month": "2026-08",
        "release_date": "2026-09-01",
        "point_estimate": 54.9,
        "p_above_consensus": 0.38,
        "confidence": "medium",
        "arguments_for_higher": [_arg()],
        "arguments_for_lower": [_arg(direction="down", url="https://example.org/b")],
        "key_uncertainty": "Tariff pass-through timing.",
        "discarded_after_cutoff": [],
    }
    base.update(over)
    return base


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR {name}: {type(e).__name__}: {e}")
        return False
    print(f"ok    {name}")
    return True


def t_indicator_lookup():
    ind = get_indicator(KEY)
    assert ind["unit_hint"] == "index", ind
    assert ind["variable_group"] == "core_macroeconomic_conditions", ind


def t_plain_json():
    parsed, notes = _parse_evidence_output(json.dumps(_obj()), KEY, min_arguments=2)
    assert parsed["point_estimate"] == 54.9
    assert parsed["n_args_up"] == 1 and parsed["n_args_down"] == 1
    assert "p_above=0.38" in notes, notes


def t_fenced_json():
    raw = "Here you go:\n```json\n" + json.dumps(_obj()) + "\n```\nHope that helps."
    parsed, _ = _parse_evidence_output(raw, KEY, min_arguments=2)
    assert parsed["confidence"] == "medium"


def t_prose_wrapped():
    raw = "Some preamble.\n" + json.dumps(_obj()) + "\nSome trailing note."
    parsed, _ = _parse_evidence_output(raw, KEY, min_arguments=2)
    assert parsed["point_estimate"] == 54.9


def t_direction_filled_from_list():
    arg = _arg()
    del arg["direction"]
    parsed, _ = _parse_evidence_output(
        json.dumps(_obj(arguments_for_higher=[arg])), KEY, min_arguments=2
    )
    assert parsed["arguments"][0]["direction"] == "up"


def _expect_parse_error(obj, fragment, min_arguments=2):
    try:
        _parse_evidence_output(json.dumps(obj), KEY, min_arguments=min_arguments)
    except EvidenceParseError as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected EvidenceParseError containing {fragment!r}")


def t_rejects_bad_probability():
    _expect_parse_error(_obj(p_above_consensus=1.4), "p_above_consensus")


def t_rejects_missing_source_url():
    bad = _arg()
    del bad["source_url"]
    _expect_parse_error(_obj(arguments_for_higher=[bad]), "missing fields")


def t_rejects_bad_date():
    _expect_parse_error(
        _obj(arguments_for_higher=[_arg(date="late August")]), "published_date"
    )


def t_rejects_bad_confidence():
    _expect_parse_error(_obj(confidence="pretty sure"), "confidence")


def t_rejects_wrong_indicator():
    _expect_parse_error(_obj(indicator="cpi_index"), "indicator mismatch")


def t_rejects_too_few_arguments():
    _expect_parse_error(
        _obj(arguments_for_higher=[], arguments_for_lower=[_arg(direction="down")]),
        "too few arguments",
        min_arguments=3,
    )


def t_rejects_non_json():
    try:
        _parse_evidence_output("ism_manufacturing_index=54.9", KEY)
    except EvidenceParseError:
        return
    raise AssertionError("expected EvidenceParseError on a key=value line")


def t_direction_consistency():
    parsed, _ = _parse_evidence_output(
        json.dumps(_obj(point_estimate=56.0, p_above_consensus=0.3)), KEY, min_arguments=2
    )
    warning = _check_direction_consistency(parsed, 55.2)
    assert "inconsistent" in warning, warning

    parsed_ok, _ = _parse_evidence_output(json.dumps(_obj()), KEY, min_arguments=2)
    assert _check_direction_consistency(parsed_ok, 55.2) == ""


def t_leak_audit_flags_post_cutoff():
    obj = _obj(
        arguments_for_higher=[_arg(date="2026-08-25")],
        arguments_for_lower=[_arg(direction="down", url="https://leak.example/x", date="2026-09-01")],
    )
    parsed, _ = _parse_evidence_output(json.dumps(obj), KEY, min_arguments=2)
    leak = audit_leakage(parsed, "2026-09-01T09:59:00-04:00")
    assert leak["leak_count"] == 1, leak
    assert leak["leaked_sources"][0]["source_url"] == "https://leak.example/x"


def t_leak_audit_clean():
    parsed, _ = _parse_evidence_output(json.dumps(_obj()), KEY, min_arguments=2)
    leak = audit_leakage(parsed, "2026-09-01T09:59:00-04:00")
    assert leak["leak_count"] == 0, leak


def t_leak_audit_noop_without_cutoff():
    parsed, _ = _parse_evidence_output(json.dumps(_obj()), KEY, min_arguments=2)
    assert audit_leakage(parsed, None)["leak_count"] == 0


def main():
    tests = [(k[2:], v) for k, v in sorted(globals().items()) if k.startswith("t_")]
    results = [check(name, fn) for name, fn in tests]
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
