"""Focused branch tests for the finite-domain precedence certificate."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from main.revision.audit_priority_collapse import Aggregate, _precedence_certificate
except ModuleNotFoundError:  # portable reproducibility-archive layout
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_priority_collapse import Aggregate, _precedence_certificate


def _aggregate(records: list[tuple[int, int, int]]) -> Aggregate:
    counts = {(mask, action): count for mask, action, count in records}
    return Aggregate(counts=counts, attempted=sum(counts.values()))


def test_precedence_certificate_acyclic_unique_chain() -> None:
    records = []
    for action in range(6):
        mask = sum(1 << channel for channel in range(action, 6))
        records.append((mask, action, action + 1))

    certificate = _precedence_certificate(_aggregate(records))

    assert certificate["acyclic"] is True
    assert certificate["unique_topological_order"] is True
    assert certificate["topological_order_high_to_low"] == [0, 1, 2, 3, 4, 5]
    assert certificate["attempted_records"] == 21
    assert certificate["edge_count"] == 15


def test_precedence_certificate_acyclic_nonunique_partial_order() -> None:
    certificate = _precedence_certificate(_aggregate([(0b000011, 0, 3)]))

    assert certificate["acyclic"] is True
    assert certificate["unique_topological_order"] is False
    assert certificate["topological_order_high_to_low"] is not None
    assert certificate["edge_count"] == 1


def test_precedence_certificate_rejects_cycle() -> None:
    certificate = _precedence_certificate(
        _aggregate([(0b000011, 0, 1), (0b000011, 1, 1)])
    )

    assert certificate["acyclic"] is False
    assert certificate["unique_topological_order"] is False
    assert certificate["topological_order_high_to_low"] is None
