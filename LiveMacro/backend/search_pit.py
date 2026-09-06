"""
Point-in-time (PIT) web search.

The evidence mode's leakage audit is detection: it notices when a model cited a
source published after the cutoff, but cannot stop it happening, because the
built-in WebSearch of GPT-5 and Claude has no date filter. This module is the
prevention layer — a search function that physically cannot return post-cutoff
documents, so a backtest agent never sees the answer in the first place.

Two guards, deliberately redundant:
  1. the provider's own date filter (Exa's endPublishedDate, Tavily's end_date);
  2. a local re-check of every returned document's date, because provider metadata
     is missing or wrong often enough that trusting it alone would be careless.

Anything without a usable publication date is dropped by default: an undated
document cannot be proven to predate the cutoff, so it is not admissible.

Providers:
  exa     — primary. Real endPublishedDate filtering, returns publishedDate.
  tavily  — fallback. Weaker date support; local filtering carries more of the load.

Environment:
  EXA_API_KEY / TAVILY_API_KEY
  PIT_SEARCH_PROVIDER   force a provider (default: exa if keyed, else tavily)
"""

import datetime as dt
import os
from urllib.parse import urlparse

import requests

from config import get_logger

logger = get_logger(__name__)

EXA_ENDPOINT = "https://api.exa.ai/search"
EXA_CONTENTS_ENDPOINT = "https://api.exa.ai/contents"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"

DEFAULT_NUM_RESULTS = 10
REQUEST_TIMEOUT_SECS = 45


class PitSearchError(RuntimeError):
    """Search could not be performed as specified. Never degrade to unfiltered search."""


# ---------- date handling ----------
def parse_cutoff(as_of):
    """
    Accept '2026-09-01', '2026-09-01T09:59:00-04:00' or a datetime.
    Returns a timezone-aware UTC datetime.
    """
    if isinstance(as_of, dt.datetime):
        parsed = as_of
    else:
        text = str(as_of).strip()
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = dt.datetime.combine(dt.date.fromisoformat(text[:10]), dt.time.min)
            except ValueError as e:
                raise PitSearchError(f"Unparseable cutoff: {as_of!r}") from e

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_published(value):
    """Parse a provider's publication date into an aware UTC datetime, or None."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text[:19], text[:10]):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    return None


def _is_admissible(published, cutoff, allow_same_day):
    """
    Admissible means provably before the cutoff.

    Same-day documents are excluded by default. A release at 10:00 ET and a preview
    published at 08:00 the same morning often carry a date-only timestamp, so they
    are indistinguishable from the release write-up itself. Losing a few legitimate
    morning previews is a cheap price for not silently importing the answer.
    """
    if published is None:
        return False, "no publication date"
    if published >= cutoff:
        return False, f"published {published.date()} >= cutoff {cutoff.date()}"
    if not allow_same_day and published.date() == cutoff.date():
        return False, f"same-day as cutoff ({cutoff.date()}) and same-day not allowed"
    return True, ""


# ---------- provider selection ----------
def _api_key(provider):
    return os.getenv("EXA_API_KEY" if provider == "exa" else "TAVILY_API_KEY", "").strip()


def resolve_provider(provider=None):
    chosen = (provider or os.getenv("PIT_SEARCH_PROVIDER") or "").strip().lower()
    if chosen:
        if chosen not in ("exa", "tavily"):
            raise PitSearchError(f"Unknown PIT search provider: {chosen!r}")
        if not _api_key(chosen):
            raise PitSearchError(
                f"Provider {chosen!r} selected but {'EXA_API_KEY' if chosen == 'exa' else 'TAVILY_API_KEY'} is not set"
            )
        return chosen

    if _api_key("exa"):
        return "exa"
    if _api_key("tavily"):
        logger.warning("EXA_API_KEY not set; falling back to Tavily, whose date filtering is weaker")
        return "tavily"
    raise PitSearchError(
        "No point-in-time search provider configured. Set EXA_API_KEY (preferred) or TAVILY_API_KEY."
    )


# ---------- providers ----------
def _post(url, *, headers=None, json_body=None):
    try:
        response = requests.post(url, headers=headers or {}, json=json_body, timeout=REQUEST_TIMEOUT_SECS)
    except requests.RequestException as e:
        raise PitSearchError(f"{urlparse(url).netloc} request failed: {e}") from e
    if response.status_code >= 400:
        raise PitSearchError(f"{urlparse(url).netloc} returned {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as e:
        raise PitSearchError(f"{urlparse(url).netloc} returned non-JSON: {response.text[:200]}") from e


def _search_exa(query, cutoff, num_results, include_text):
    body = {
        "query": query,
        "numResults": num_results,
        "endPublishedDate": cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "type": "auto",
    }
    if include_text:
        body["contents"] = {"text": {"maxCharacters": 2000}}

    payload = _post(EXA_ENDPOINT, headers={"x-api-key": _api_key("exa")}, json_body=body)
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "published_raw": item.get("publishedDate"),
            "author": item.get("author") or "",
            "text": (item.get("text") or "")[:2000],
        }
        for item in payload.get("results", [])
    ]


def _search_tavily(query, cutoff, num_results, include_text):
    body = {
        "api_key": _api_key("tavily"),
        "query": query,
        "max_results": num_results,
        "include_raw_content": bool(include_text),
        # Newer Tavily builds honour end_date; older ones ignore it, which is why
        # the local re-check below is not optional.
        "end_date": cutoff.strftime("%Y-%m-%d"),
    }
    payload = _post(TAVILY_ENDPOINT, json_body=body)
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "published_raw": item.get("published_date") or item.get("published_time"),
            "author": "",
            "text": (item.get("raw_content") or item.get("content") or "")[:2000],
        }
        for item in payload.get("results", [])
    ]


# ---------- public API ----------
def pit_search(
    query,
    as_of,
    num_results=DEFAULT_NUM_RESULTS,
    provider=None,
    include_text=True,
    allow_same_day=False,
    require_date=True,
):
    """
    Search the web as it stood before `as_of`.

    Returns {"provider", "cutoff", "query", "results", "rejected", "counts"}.
    `results` holds only documents proven to predate the cutoff; `rejected` records
    what was dropped and why, so a run can show its own filtering worked.
    """
    cutoff = parse_cutoff(as_of)
    chosen = resolve_provider(provider)
    fetch = _search_exa if chosen == "exa" else _search_tavily

    raw_results = fetch(query, cutoff, num_results, include_text)

    admitted, rejected = [], []
    for item in raw_results:
        published = parse_published(item.get("published_raw"))
        ok, reason = _is_admissible(published, cutoff, allow_same_day)
        if not ok and published is None and not require_date:
            ok, reason = True, ""
        record = {
            "title": item["title"],
            "url": item["url"],
            "published_date": published.date().isoformat() if published else None,
            "author": item["author"],
            "text": item["text"],
        }
        if ok:
            admitted.append(record)
        else:
            rejected.append({**record, "reason": reason})

    if rejected:
        logger.info(
            "PIT search dropped %d/%d results for query=%r (cutoff %s)",
            len(rejected), len(raw_results), query, cutoff.isoformat(),
        )

    return {
        "provider": chosen,
        "cutoff": cutoff.isoformat(),
        "query": query,
        "results": admitted,
        "rejected": rejected,
        "counts": {
            "returned_by_provider": len(raw_results),
            "admitted": len(admitted),
            "rejected": len(rejected),
        },
    }


def pit_get_contents(url, as_of, provider=None, max_characters=6000, allow_same_day=False):
    """
    Fetch one document's text, but only if it predates the cutoff.

    A separate check matters: an agent can reach a post-cutoff page by following a
    URL it saw in an admitted result, so the gate has to sit on fetching too.
    """
    cutoff = parse_cutoff(as_of)
    chosen = resolve_provider(provider)

    if chosen == "exa":
        payload = _post(
            EXA_CONTENTS_ENDPOINT,
            headers={"x-api-key": _api_key("exa")},
            json_body={"urls": [url], "text": {"maxCharacters": max_characters}},
        )
        items = payload.get("results", [])
        if not items:
            raise PitSearchError(f"No content returned for {url}")
        item = items[0]
        published_raw, text, title = item.get("publishedDate"), item.get("text") or "", item.get("title") or ""
    else:
        payload = _post(
            TAVILY_EXTRACT_ENDPOINT,
            json_body={"api_key": _api_key("tavily"), "urls": [url]},
        )
        items = payload.get("results", [])
        if not items:
            raise PitSearchError(f"No content returned for {url}")
        item = items[0]
        published_raw = item.get("published_date")
        text, title = (item.get("raw_content") or "")[:max_characters], ""

    published = parse_published(published_raw)
    ok, reason = _is_admissible(published, cutoff, allow_same_day)
    if not ok:
        logger.warning("PIT contents BLOCKED for %s: %s", url, reason)
        return {
            "url": url,
            "blocked": True,
            "reason": reason,
            "published_date": published.date().isoformat() if published else None,
            "text": "",
        }

    return {
        "url": url,
        "blocked": False,
        "reason": "",
        "title": title,
        "published_date": published.date().isoformat(),
        "text": text[:max_characters],
    }
