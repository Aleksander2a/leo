"""Locating data files that ship beside the code, in either layout.

A source checkout keeps `resources/`, `migrations/`, and `evals/` at the
repository root, two levels above `src/leo/`. An installed package puts the code
in site-packages and those directories somewhere else entirely -- the container
copies them next to the app instead. Deriving a path from a module's own
position encodes the checkout layout and silently produces nonsense everywhere
else: the deployed worker resolved its skill catalogue to
`/usr/local/lib/python3.12/resources/leo-skills`, and because `glob` on a
missing directory returns an empty list rather than raising, it ran with no
skills at all and said nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# `src/leo/<module>.py` -> repository root is three parents up; one spare.
_ANCHOR_SEARCH_DEPTH = 4


def find_data_directory(relative: str, *, anchor: Path | None = None) -> Path | None:
    """Return the real directory for a packaged path, or None if it is absent.

    The working directory is tried first, because that is where a container
    places these files, then each parent of the anchoring module. Returning None
    is meaningful: the caller can decide whether the data is required, rather
    than silently behaving as though it were present but empty.
    """

    source = anchor or Path(__file__)
    # Bounded: `src/leo/<module>.py` reaches the repository root three levels up,
    # so a handful covers every real layout. An unbounded walk would climb to the
    # filesystem root and could adopt an unrelated directory that happens to
    # match somewhere above the application.
    nearby = source.resolve().parents[:_ANCHOR_SEARCH_DEPTH]
    for candidate in (Path.cwd(), *nearby):
        located = candidate / relative
        if located.is_dir():
            return located
    return None


def require_data_directory(relative: str, *, anchor: Path | None = None) -> Path:
    """Locate a packaged directory, warning loudly when it is missing.

    The returned path is still usable for callers that degrade gracefully, but
    the absence is recorded once at startup so a deployment missing its data
    shows up in the logs instead of as quietly reduced behaviour.
    """

    located = find_data_directory(relative, anchor=anchor)
    if located is not None:
        return located
    logger.warning(
        "packaged data directory %r was not found next to the application or the "
        "installed package; features that depend on it are unavailable",
        relative,
    )
    return Path(relative)
