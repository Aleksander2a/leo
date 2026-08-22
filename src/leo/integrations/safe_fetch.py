"""Bounded public HTTP fetch policy for untrusted research material."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx


class FetchPolicyError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    max_bytes: int = 32_768
    max_redirects: int = 3
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class FetchedArtifact:
    requested_url: str
    final_url: str
    redirect_count: int
    content_type: str
    text: str
    sha256: str
    byte_count: int
    truncated: bool
    peer_ip: str
    dns_pin_sha256: str
    untrusted: bool = True


@dataclass(frozen=True, slots=True)
class PublicUrlResolution:
    host: str
    addresses: tuple[str, ...]
    fingerprint: str


def validate_public_url(url: str) -> PublicUrlResolution:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FetchPolicyError("fetch_url_scheme_denied")
    if parsed.username or parsed.password:
        raise FetchPolicyError("fetch_url_credentials_denied")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise FetchPolicyError("fetch_private_host_denied")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, None)
            ]
        except OSError as exc:
            raise FetchPolicyError("fetch_dns_failed") from exc
    if any(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        for address in addresses
    ):
        raise FetchPolicyError("fetch_private_host_denied")
    normalized = tuple(sorted({_normalized_ip(address) for address in addresses}))
    if not normalized:
        raise FetchPolicyError("fetch_dns_failed")
    encoded = "\n".join(normalized).encode("ascii")
    return PublicUrlResolution(
        host=host,
        addresses=normalized,
        fingerprint=hashlib.sha256(encoded).hexdigest(),
    )


async def fetch_public_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: FetchPolicy | None = None,
) -> FetchedArtifact:
    policy = policy or FetchPolicy()
    if policy.max_bytes < 1 or policy.max_redirects < 0 or policy.timeout_seconds <= 0:
        raise FetchPolicyError("fetch_policy_invalid")
    requested_url = url
    redirect_count = 0
    while True:
        resolution = validate_public_url(url)
        try:
            async with client.stream(
                "GET",
                url,
                follow_redirects=False,
                timeout=policy.timeout_seconds,
            ) as response:
                peer_ip = _validated_peer_ip(response, resolution)
                if response.is_redirect:
                    if redirect_count >= policy.max_redirects:
                        raise FetchPolicyError("fetch_redirect_limit")
                    location = response.headers.get("location")
                    if not location:
                        raise FetchPolicyError("fetch_redirect_missing_location")
                    url = urljoin(url, location)
                    redirect_count += 1
                    continue
                if response.status_code == 429:
                    raise FetchPolicyError("fetch_rate_limited")
                if response.status_code >= 500:
                    raise FetchPolicyError("fetch_upstream_unavailable")
                if response.status_code >= 400:
                    raise FetchPolicyError("fetch_request_rejected")
                content_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                if content_type not in {"text/plain", "text/html", "application/json"}:
                    raise FetchPolicyError("fetch_content_type_denied")
                buffered = bytearray()
                async for chunk in response.aiter_bytes():
                    remaining = policy.max_bytes + 1 - len(buffered)
                    if remaining <= 0:
                        break
                    buffered.extend(chunk[:remaining])
                    if len(buffered) > policy.max_bytes:
                        break
        except FetchPolicyError:
            raise
        except httpx.TimeoutException as exc:
            raise FetchPolicyError("fetch_timeout") from exc
        except httpx.TransportError as exc:
            raise FetchPolicyError("fetch_transport_error") from exc
        truncated = len(buffered) > policy.max_bytes
        raw = bytes(buffered[: policy.max_bytes])
        text = _sanitize_untrusted_text(raw.decode("utf-8", errors="replace"), content_type)
        if not text:
            raise FetchPolicyError("fetch_empty_content")
        return FetchedArtifact(
            requested_url=requested_url,
            final_url=url,
            redirect_count=redirect_count,
            content_type=content_type,
            text=text,
            sha256=hashlib.sha256(raw).hexdigest(),
            byte_count=len(raw),
            truncated=truncated,
            peer_ip=peer_ip,
            dns_pin_sha256=resolution.fingerprint,
        )


def _validated_peer_ip(
    response: httpx.Response,
    resolution: PublicUrlResolution,
) -> str:
    """Fail closed unless the connected peer is one of the pre-resolved public IPs."""

    raw_peer: object = response.extensions.get("leo_peer_ip")
    if raw_peer is None:
        stream = response.extensions.get("network_stream")
        get_extra_info = getattr(stream, "get_extra_info", None)
        if callable(get_extra_info):
            raw_peer = get_extra_info("server_addr")
    if isinstance(raw_peer, (tuple, list)) and raw_peer:
        raw_peer = raw_peer[0]
    if not isinstance(raw_peer, str):
        raise FetchPolicyError("fetch_peer_unverifiable")
    try:
        peer = ipaddress.ip_address(raw_peer)
    except ValueError as exc:
        raise FetchPolicyError("fetch_peer_unverifiable") from exc
    normalized = _normalized_ip(peer)
    if normalized not in resolution.addresses:
        raise FetchPolicyError("fetch_dns_rebinding_detected")
    return normalized


def _normalized_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(address)


def _sanitize_untrusted_text(text: str, content_type: str) -> str:
    if content_type == "text/html":
        parser = _UntrustedHTMLTextParser()
        parser.feed(text)
        parser.close()
        text = " ".join(parser.text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return " ".join(text.split())


class _UntrustedHTMLTextParser(HTMLParser):
    _ACTIVE_TAGS = frozenset({"script", "style", "iframe", "object", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self._blocked_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag.lower() in self._ACTIVE_TAGS:
            self._blocked_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ACTIVE_TAGS and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth:
            self.text.append(data)
