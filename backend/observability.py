"""Making the active matting configuration visible in the logs.

A 512MB host only survives with `MATTE_SOLVE_MODE=tiled`, and the mode is an environment
variable, so "which mode is this deployment actually running?" has to be answerable from
the log stream rather than from the deploy dashboard.
"""

from __future__ import annotations

import logging
import os
import sys

from backend.jobs.gate import DEFAULT_MAX_WORKERS, max_workers
from backend.lash_extraction.matting import solve_settings

logger = logging.getLogger("backend.matte")
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging() -> None:
    """Give the `backend` loggers a stdout handler, at `MATTE_LOG_LEVEL` (default INFO).

    uvicorn configures its own loggers and leaves the root logger without a handler, so
    without this the matte lines would be dropped on a real deployment - which is exactly
    where they are needed. Records still propagate so a host that configures logging itself
    (pytest, gunicorn) keeps seeing them.
    """
    backend_logger = logging.getLogger("backend")
    requested = os.environ.get("MATTE_LOG_LEVEL", "").strip().upper()
    # An unknown level falls back to INFO: a typo here must not silence the very lines that
    # tell an operator which mode the deployment is running.
    backend_logger.setLevel(logging.getLevelNamesMapping().get(requested, logging.INFO))
    if not backend_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        backend_logger.addHandler(handler)


def _describe(value: int | None) -> str:
    return "none" if value is None else str(value)


def log_matte_settings() -> None:
    """One startup line naming the mode, the pixel budget and the concurrency limit.

    A misconfigured value is logged as an error instead of propagating: crashing the
    import would take the whole app down with a traceback that hides the actual typo.
    """
    configure_logging()
    try:
        mode, budget = solve_settings()
        workers = max_workers()
    except ValueError as exc:
        logger.error("matte settings are misconfigured: %s", exc)
        return
    defaults = mode == "full" and budget is None and workers == DEFAULT_MAX_WORKERS
    logger.info(
        "matte settings: solve_mode=%s max_solve_pixels=%s max_workers=%d%s",
        mode,
        _describe(budget),
        workers,
        " (all default; set MATTE_SOLVE_MODE=tiled on 512MB hosts)" if defaults else "",
    )
