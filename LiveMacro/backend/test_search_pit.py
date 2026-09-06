"""
Offline checks for the point-in-time search layer. Provider HTTP is stubbed, so
this makes no network calls and needs no API key.

    python test_search_pit.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import search_pit
from search_pit import (
    PitSearchError,
    _date_from_text,
    _date_from_url,
    parse_cutoff,
    parse_published,
    pit_get_contents,
    pit_search,
    resolve_published,
)

CUTOFF = "2026-09-01T09:59:00-04:00"  # 13:59 UTC


class StubPost:
    """Replaces search_pit._post for the duration of a test."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self._saved = None

    def __enter__(self):
        self._saved = search_pit._post
        outer = self

        def _fake(url, *, headers=None, json_body=None):
            outer.calls.append({"url": url, "headers": headers, "body": json_body})
            return outer.payload

        search_pit._post = _fake
        return self

    def __exit__(self, *exc):
        search_pit._post = self._saved
        return False


def _exa_payload(results):
    return {"results": results}


def _hit(url, published, title="t", text="body"):
    return {"url": url, "publishedDate": published, "title": title, "text": text, "author": ""}


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


# ---------- date primitives ----------
def t_parse_cutoff_forms():
    a = parse_cutoff("2026-09-01T09:59:00-04:00")
    assert (a.hour, a.minute) == (13, 59), a
    b = parse_cutoff("2026-09-01")
    assert b.date().isoformat() == "2026-09-01"
    try:
        parse_cutoff("not a date")
    except PitSearchError:
        return
    raise AssertionError("expected PitSearchError on garbage cutoff")


def t_parse_published_forms():
    assert parse_published("2026-08-24").date().isoformat() == "2026-08-24"
    assert parse_published("2026-08-24T11:00:00.000Z").date().isoformat() == "2026-08-24"
    assert parse_published(None) is None
    assert parse_published("sometime in August") is None


# ---------- provider selection ----------
def _with_env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def t_provider_requires_a_key():
    saved = _with_env(EXA_API_KEY=None, TAVILY_API_KEY=None, PIT_SEARCH_PROVIDER=None)
    try:
        search_pit.resolve_provider()
    except PitSearchError as e:
        assert "No point-in-time search provider" in str(e)
        return
    finally:
        _restore(saved)
    raise AssertionError("expected PitSearchError when no key is set")


def t_provider_prefers_exa():
    saved = _with_env(EXA_API_KEY="x", TAVILY_API_KEY="y", PIT_SEARCH_PROVIDER=None)
    try:
        assert search_pit.resolve_provider() == "exa"
    finally:
        _restore(saved)


def t_forced_provider_without_key_fails():
    saved = _with_env(EXA_API_KEY=None, TAVILY_API_KEY=None, PIT_SEARCH_PROVIDER="exa")
    try:
        search_pit.resolve_provider()
    except PitSearchError as e:
        assert "EXA_API_KEY" in str(e)
        return
    finally:
        _restore(saved)
    raise AssertionError("expected PitSearchError when the forced provider has no key")


# ---------- filtering: the point of the module ----------
def t_sends_cutoff_to_provider():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        with StubPost(_exa_payload([])) as stub:
            pit_search("q", as_of=CUTOFF)
        body = stub.calls[0]["body"]
        assert body["endPublishedDate"] == "2026-09-01T13:59:00.000Z", body
    finally:
        _restore(saved)


def t_drops_post_cutoff_even_if_provider_returns_it():
    """The provider filter is not trusted on its own — this is the whole design."""
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([
            _hit("https://ok.example/a", "2026-08-24"),
            _hit("https://leak.example/answer", "2026-09-01T14:05:00.000Z"),
            _hit("https://leak.example/later", "2026-09-03"),
        ])
        with StubPost(payload):
            result = pit_search("q", as_of=CUTOFF)
        urls = [r["url"] for r in result["results"]]
        assert urls == ["https://ok.example/a"], urls
        assert result["counts"] == {"returned_by_provider": 3, "admitted": 1, "rejected": 2}
        assert all("leak.example" in r["url"] for r in result["rejected"])
    finally:
        _restore(saved)


def t_drops_undated_by_default():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        with StubPost(_exa_payload([_hit("https://nodate.example/a", None)])):
            result = pit_search("q", as_of=CUTOFF)
        assert result["counts"]["admitted"] == 0
        assert result["rejected"][0]["reason"] == "no publication date"

        with StubPost(_exa_payload([_hit("https://nodate.example/a", None)])):
            lenient = pit_search("q", as_of=CUTOFF, require_date=False)
        assert lenient["counts"]["admitted"] == 1
    finally:
        _restore(saved)


def t_same_day_excluded_by_default():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([_hit("https://sameday.example/a", "2026-09-01")])
        with StubPost(payload):
            strict = pit_search("q", as_of=CUTOFF)
        assert strict["counts"]["admitted"] == 0, strict["counts"]
        with StubPost(payload):
            lenient = pit_search("q", as_of=CUTOFF, allow_same_day=True)
        assert lenient["counts"]["admitted"] == 1, lenient["counts"]

        # ...but a same-day document timestamped after the cutoff stays blocked
        # whatever the flag says: allow_same_day relaxes the date rule, never the clock.
        after = _exa_payload([_hit("https://sameday.example/b", "2026-09-01T14:05:00.000Z")])
        with StubPost(after):
            still_blocked = pit_search("q", as_of=CUTOFF, allow_same_day=True)
        assert still_blocked["counts"]["admitted"] == 0, still_blocked["counts"]
    finally:
        _restore(saved)


def t_contents_blocks_post_cutoff():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([
            {"url": "https://leak.example/answer", "publishedDate": "2026-09-01T14:05:00.000Z",
             "text": "ISM came in at 54.6", "title": "ISM report"}
        ])
        with StubPost(payload):
            result = pit_get_contents("https://leak.example/answer", as_of=CUTOFF)
        assert result["blocked"] is True, result
        assert result["text"] == "", "blocked content must not leak the body"
    finally:
        _restore(saved)


def t_contents_allows_pre_cutoff():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([
            {"url": "https://ok.example/a", "publishedDate": "2026-08-24",
             "text": "flash PMI fell to 53.2", "title": "S&P Global"}
        ])
        with StubPost(payload):
            result = pit_get_contents("https://ok.example/a", as_of=CUTOFF)
        assert result["blocked"] is False
        assert "53.2" in result["text"]
    finally:
        _restore(saved)


# ---------- date recovery ----------
def t_date_from_url_patterns():
    assert _date_from_url("https://x.example/2026/08/21/story").date().isoformat() == "2026-08-21"
    assert _date_from_url("https://x.example/news?date=2026-08-21").date().isoformat() == "2026-08-21"
    assert _date_from_url("https://x.example/report_20260821_final.html").date().isoformat() == "2026-08-21"
    assert _date_from_url("https://x.example/Public/PressRelease/552d682e") is None
    assert _date_from_url("https://x.example/2026/13/45/impossible") is None


def t_date_from_text_takes_latest():
    """A press release names the month it covers before its own dateline."""
    text = "August 2026 data. NEW YORK, August 21, 2026 - S&P Global released..."
    assert _date_from_text(text).date().isoformat() == "2026-08-21"
    assert _date_from_text("21 August 2026 dateline").date().isoformat() == "2026-08-21"
    assert _date_from_text("published 2026-08-21 by us").date().isoformat() == "2026-08-21"
    assert _date_from_text("no dates at all here") is None


def t_dateline_recovery_only_for_known_publishers():
    text = "NEW YORK, August 21, 2026 - the report says..."
    known, source = resolve_published(None, "https://www.pmi.spglobal.com/Public/x/GUID", text)
    assert source == "dateline" and known.date().isoformat() == "2026-08-21"

    unknown, source = resolve_published(None, "https://randomblog.example/post", text)
    assert unknown is None and source == "none", "an arbitrary page's first date is not its dateline"


def t_provider_date_wins_over_recovery():
    provider = parse_published("2026-08-10")
    got, source = resolve_published(provider, "https://x.example/2026/08/21/story", "August 25, 2026")
    assert source == "provider" and got.date().isoformat() == "2026-08-10"


def t_recovery_cannot_admit_a_post_cutoff_document():
    """Recovery takes the latest plausible date, so a mis-read blocks rather than admits."""
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([
            {"url": "https://www.pmi.spglobal.com/Public/x/GUID", "publishedDate": None, "author": "",
             "title": "release", "text": "Covering August 2026. NEW YORK, September 2, 2026 - results..."}
        ])
        with StubPost(payload):
            result = pit_search("q", as_of=CUTOFF)
        assert result["counts"]["admitted"] == 0, result["counts"]
        assert "2026-09-02" in result["rejected"][0]["reason"]
    finally:
        _restore(saved)


def t_recovery_rescues_undated_primary_source():
    saved = _with_env(EXA_API_KEY="k", PIT_SEARCH_PROVIDER="exa")
    try:
        payload = _exa_payload([
            {"url": "https://www.pmi.spglobal.com/Public/x/GUID", "publishedDate": None, "author": "",
             "title": "Flash PMI", "text": "NEW YORK, August 21, 2026 - flash manufacturing PMI fell to 53.2."}
        ])
        with StubPost(payload):
            result = pit_search("q", as_of=CUTOFF)
        assert result["counts"]["admitted"] == 1, result["counts"]
        assert result["results"][0]["date_source"] == "dateline"
    finally:
        _restore(saved)


# ---------- agent wiring ----------
def t_agent_denies_builtin_web_tools():
    import llm_clients.claude_code_agent_pit as pit

    assert "WebSearch" in pit.DISALLOWED_TOOLS and "WebFetch" in pit.DISALLOWED_TOOLS
    assert pit.ALLOWED_TOOLS == ["mcp__pit__search", "mcp__pit__get_contents"]

    allow = asyncio.run(pit._deny_non_pit_tools({"tool_name": "mcp__pit__search"}, None, None))
    assert allow["hookSpecificOutput"]["permissionDecision"] == "allow"

    for blocked in ("WebSearch", "WebFetch", "Bash"):
        deny = asyncio.run(pit._deny_non_pit_tools({"tool_name": blocked}, None, None))
        assert deny["hookSpecificOutput"]["permissionDecision"] == "deny", blocked


def t_agent_requires_as_of():
    from llm_clients import get_client

    try:
        get_client("claude-code-agent-pit")("sys", "user")
    except ValueError as e:
        assert "as_of" in str(e)
        return
    raise AssertionError("expected ValueError when as_of is missing")


def t_evidence_job_wiring_rejects_missing_cutoff():
    from forecasting_evidence import forecast_evidence_once

    job = {
        "id": "no_cutoff",
        "indicator": "ism_manufacturing_index",
        "target_period": "2026-08",
        "release_date": "2026-09-01",
    }
    try:
        forecast_evidence_once(job, "claude-code-agent-pit", __import__("datetime").datetime.now())
    except ValueError as e:
        assert "as_of" in str(e)
        return
    raise AssertionError("expected ValueError for a PIT job without as_of")


def main():
    tests = [(k[2:], v) for k, v in sorted(globals().items()) if k.startswith("t_")]
    results = [check(name, fn) for name, fn in tests]
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
