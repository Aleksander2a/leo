"""Small no-network URL checks for provider-returned references.

This module does not resolve hostnames and does not fetch anything.  It rejects
credentials, localhost names, non-public IP literals, and malformed/non-HTTPS
URLs before an untrusted provider URL can become a discovery result or a
user-facing source link.  The public fetch adapter still owns DNS resolution,
redirect, and peer-address enforcement at request time.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.IGNORECASE)
_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain"})


def is_public_https_url(value: str, *, max_length: int = 2_048) -> bool:
    """Return whether a URL is a literal-safe candidate for public HTTPS use.

    DNS names are deliberately not resolved here.  A later network fetch must
    independently resolve and pin the peer address under the fetch policy.
    """

    if not isinstance(value, str) or not 1 <= len(value) <= max_length:
        return False
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # Accessing port is a useful validation step: urlsplit otherwise accepts
        # non-numeric and out-of-range ports until the property is evaluated.
        _ = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False

    normalized_host = hostname.casefold().rstrip(".")
    if not normalized_host or normalized_host in _LOCAL_NAMES:
        return False
    if normalized_host.endswith((".localhost", ".localhost.localdomain")):
        return False

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        # Reject ambiguous numeric spellings (for example 127.1 or an integer
        # host) rather than letting a later URL client reinterpret them as IPs.
        if re.fullmatch(r"(?:0x[0-9a-f]+|[0-9.]+)", normalized_host):
            return False
        try:
            ascii_host = normalized_host.encode("idna").decode("ascii")
        except UnicodeError:
            return False
        labels = ascii_host.split(".")
        return len(labels) >= 2 and all(_DNS_LABEL.fullmatch(label) for label in labels)
    return address.is_global and not address.is_multicast
