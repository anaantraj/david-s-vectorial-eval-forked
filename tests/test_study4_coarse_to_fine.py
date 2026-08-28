from __future__ import annotations

from scripts.partition_study4_coarse_to_fine import partition


def test_partition_is_balanced_disjoint_and_complete():
    rows = [
        {"record_id": f"a{i}", "audience": "a"} for i in range(5)
    ] + [{"record_id": f"b{i}", "audience": "b"} for i in range(4)]
    platform, audience = partition(rows)
    platform_ids = {row["record_id"] for row in platform}
    audience_ids = {row["record_id"] for row in audience}
    assert platform_ids.isdisjoint(audience_ids)
    assert platform_ids | audience_ids == {row["record_id"] for row in rows}
    assert sum(row["audience"] == "a" for row in platform) == 3
    assert sum(row["audience"] == "a" for row in audience) == 2
    assert sum(row["audience"] == "b" for row in platform) == 2
    assert sum(row["audience"] == "b" for row in audience) == 2


def test_partition_is_input_order_independent():
    rows = [{"record_id": f"r{i}", "audience": "a"} for i in range(10)]
    left = partition(rows)
    right = partition(list(reversed(rows)))
    assert [[row["record_id"] for row in half] for half in left] == [
        [row["record_id"] for row in half] for half in right
    ]
