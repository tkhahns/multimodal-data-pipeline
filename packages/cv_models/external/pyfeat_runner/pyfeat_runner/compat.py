"""Compatibility helpers for SciPy-dependent features."""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

_APPLIED = False


def _normalize_binom_args(x, n) -> Tuple[int, int]:
    if n is None:
        if isinstance(x, (int, float)):
            raise ValueError("When n is None, x must provide successes and failures.")
        if not isinstance(x, Iterable):
            raise ValueError("Iterable required when n is None.")
        items = list(x)
        if len(items) != 2:
            raise ValueError("Expected length-2 sequence when n is None.")
        successes = items[0]
        failures = items[1]
        total = successes + failures
        return int(successes), int(total)
    return int(x), int(n)


def ensure_legacy_stats() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        import scipy
        stats = scipy.stats
    except Exception as exc:  # pragma: no cover
        logger.debug("SciPy unavailable in runner; skipping legacy patch: %s", exc)
        _APPLIED = True
        return

    if hasattr(stats, "binom_test"):
        _APPLIED = True
        return

    try:
        from scipy.stats import binomtest
    except Exception as exc:  # pragma: no cover
        logger.warning("SciPy>=1.11 detected but binomtest missing; cannot patch binom_test: %s", exc)
        _APPLIED = True
        return

    def _binom_test(x, n=None, p=0.5, alternative="two-sided") -> float:
        successes, trials = _normalize_binom_args(x, n)
        return float(binomtest(successes, n=trials, p=p, alternative=alternative).pvalue)

    stats.binom_test = _binom_test  # type: ignore[attr-defined]
    logger.info("Patched scipy.stats.binom_test inside Py-Feat runner for compatibility.")
    _APPLIED = True


__all__ = ["ensure_legacy_stats"]
