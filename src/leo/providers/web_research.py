"""Provider-neutral helpers for bounded verified-web provider families."""

from __future__ import annotations

import math
import re
from urllib.parse import urlsplit

from pydantic import JsonValue

from leo.url_policy import is_public_https_url


def rank_tavily_result_urls(results: JsonValue, objective: str) -> tuple[str, ...]:
    """Rank normalized Tavily discovery URLs with an authority preference."""

    if not isinstance(results, list):
        return ()
    objective_tokens = frozenset(re.findall(r"[a-z][a-z0-9]{2,31}", objective.casefold()))
    ranked: list[tuple[int, float, int, str]] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not (isinstance(url, str) and len(url) <= 2_048 and is_public_https_url(url)):
            continue
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        if _non_article_discovery_host(host) or parsed.path.casefold().endswith(
            (".avi", ".mov", ".mp3", ".mp4", ".pdf", ".wav", ".webm")
        ):
            continue
        host_tokens = frozenset(re.findall(r"[a-z][a-z0-9]{2,31}", host))
        authority = 0
        if host.endswith((".gov", ".edu")):
            authority += 8
        if host.startswith(("docs.", "developer.", "developers.")):
            authority += 6
        if objective_tokens.intersection(host_tokens) or any(
            len(token) >= 4 and token in host for token in objective_tokens
        ):
            authority += 5
        if re.search(r"/(?:docs?|documentation|whatsnew|releases?)(?:/|$)", parsed.path):
            authority += 3
        if host in {"dev.to", "medium.com", "reddit.com"} or host.endswith(
            (".medium.com", ".substack.com")
        ):
            authority -= 8
        raw_score = item.get("score")
        score = (
            float(raw_score)
            if (
                isinstance(raw_score, int | float)
                and not isinstance(raw_score, bool)
                and math.isfinite(float(raw_score))
                and 0 <= raw_score <= 1
            )
            else 0.0
        )
        ranked.append((authority, score, index, url))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(dict.fromkeys(item[3] for item in ranked))


def valid_verified_web_attempts(data: dict[str, JsonValue]) -> bool:
    attempts = data.get("provider_attempts")
    attempt_count = data.get("provider_attempt_count")
    if not (
        data.get("provider_family_version") == "verified-web-provider-family-v1"
        and isinstance(attempts, list)
        and 1 <= len(attempts) <= 3
        and isinstance(attempt_count, int)
        and not isinstance(attempt_count, bool)
        and attempt_count == len(attempts)
    ):
        return False
    for attempt in attempts:
        if not isinstance(attempt, dict):
            return False
        provider = attempt.get("provider")
        stage = attempt.get("stage")
        status = attempt.get("status")
        code = attempt.get("code")
        if not (
            provider in {"exa", "tavily", "public-web"}
            and stage in {"search", "fetch"}
            and status in {"failed", "succeeded"}
            and (code is None or isinstance(code, str))
        ):
            return False
        if status == "succeeded":
            if code is not None:
                return False
        elif not isinstance(code, str) or not code:
            return False
    selected = data.get("selected_provider")
    if selected == "exa":
        return attempts == [{"provider": "exa", "stage": "search", "status": "succeeded"}]
    if selected == "tavily_public_fetch":
        expected_tail = [
            {"provider": "tavily", "stage": "search", "status": "succeeded"},
            {"provider": "public-web", "stage": "fetch", "status": "succeeded"},
        ]
        if attempts == expected_tail:
            return True
        return (
            len(attempts) == 3
            and isinstance(attempts[0], dict)
            and attempts[0].get("provider") == "exa"
            and attempts[0].get("stage") == "search"
            and attempts[0].get("status") == "failed"
            and isinstance(attempts[0].get("code"), str)
            and attempts[1:] == expected_tail
        )
    return False


def _non_article_discovery_host(host: str) -> bool:
    excluded = (
        "facebook.com",
        "instagram.com",
        "reddit.com",
        "tiktok.com",
        "x.com",
        "youtu.be",
        "youtube.com",
    )
    return any(host == item or host.endswith(f".{item}") for item in excluded)


__all__ = ["rank_tavily_result_urls", "valid_verified_web_attempts"]
