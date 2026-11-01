"""Benchmark tooling for maintainers (repo-side, not shipped).

The contents of this package live outside the installable
``sentrysearch`` package and are excluded from the source distribution;
run them from a checkout.

Issue #68 reuses the fixed accuracy workload instead of redefining it::

    from benchmarks.accuracy_eval import (
        BENCHMARK_CLIP_PATH,
        BENCHMARK_CLIP_URL,
        FLOOR_QUERIES,
        FINE_QUERIES,
        WORKLOAD,
    )
"""
