"""Local dev entry point for the read-only dashboard API.

Async Psycopg cannot run under Windows' default ``ProactorEventLoop`` (the same constraint
``leo.cli._run_async`` and the ``leo.evals.*`` operators already work around). Uvicorn's own
``Server.run()`` wraps everything in its own ``asyncio.run()`` call before the app module is
even imported, so the compatible loop has to be selected here and ``server.serve()`` driven
directly -- setting a policy inside ``leo.api.app`` would run too late to matter.

Usage: ``uv run python scripts/run_dashboard_api.py``
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    config = uvicorn.Config("leo.api.app:app", host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        ) as runner:
            runner.run(server.serve())
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
