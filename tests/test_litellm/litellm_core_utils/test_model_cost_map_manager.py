from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import (
    ModelCostMapLoadError,
    PreparedModelCostMap,
    get_model_cost_map_source_info,
    get_required_remote_model_cost_map_snapshot,
    get_required_remote_model_cost_map_keys,
    set_required_remote_model_cost_map_snapshot,
    set_required_remote_model_cost_map_keys,
)
from litellm.litellm_core_utils.model_cost_map_manager import ModelCostMapManager


@pytest.mark.asyncio
async def test_reload_failure_retains_last_successful_map(monkeypatch):
    original_model_cost = litellm.model_cost
    previous_map = {"remote-old": {"litellm_provider": "anthropic"}}
    litellm.model_cost = previous_map
    set_required_remote_model_cost_map_keys(frozenset(previous_map))
    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_REQUIRE_REMOTE", "true")

    def fail_to_load(url: str) -> PreparedModelCostMap:
        raise ModelCostMapLoadError("unavailable")

    try:
        with pytest.raises(ModelCostMapLoadError, match="unavailable"):
            await ModelCostMapManager(load_required_remote_model_cost_map=fail_to_load).reload_model_cost_map()
        assert litellm.model_cost is previous_map
    finally:
        litellm.model_cost = original_model_cost
        set_required_remote_model_cost_map_keys(frozenset())


@pytest.mark.asyncio
async def test_periodic_reload_publishes_authoritative_catalog_update(monkeypatch):
    original_model_cost = litellm.model_cost
    original_voyage_models = set(litellm.voyage_models)
    original_snapshot = get_required_remote_model_cost_map_snapshot()
    sleep_intervals: list[float] = []
    scheduled_deadlines: list[str] = []
    litellm.model_cost = {"remote-old": {"litellm_provider": "anthropic"}}
    set_required_remote_model_cost_map_keys(frozenset({"remote-old"}))
    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_REQUIRE_REMOTE", "true")

    async def sleep_once(interval_seconds: float) -> None:
        sleep_intervals.append(interval_seconds)
        next_run_at = get_model_cost_map_source_info()["next_run_at"]
        assert isinstance(next_run_at, str)
        scheduled_deadlines.append(next_run_at)
        if len(sleep_intervals) > 1:
            raise asyncio.CancelledError

    manager = ModelCostMapManager(
        load_required_remote_model_cost_map=lambda url: PreparedModelCostMap(
            model_cost={"remote-new": {"litellm_provider": "voyage"}},
            catalog_snapshot={"remote-new": {"litellm_provider": "voyage"}},
        ),
        sleep=sleep_once,
        utc_now=lambda: datetime(2026, 8, 18, tzinfo=timezone.utc),
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            await manager.run_periodic_reload(300)
        assert sleep_intervals == [300, 300]
        assert scheduled_deadlines == [
            "2026-08-18T00:05:00+00:00",
            "2026-08-18T00:05:00+00:00",
        ]
        assert get_model_cost_map_source_info()["next_run_at"] is None
        assert "remote-old" not in litellm.model_cost
        assert "remote-new" in litellm.model_cost
        assert get_required_remote_model_cost_map_snapshot() == {
            "remote-new": {"litellm_provider": "voyage"}
        }
    finally:
        litellm.model_cost = original_model_cost
        set_required_remote_model_cost_map_snapshot(original_snapshot)
        set_required_remote_model_cost_map_keys(frozenset())
        litellm.voyage_models.clear()
        litellm.voyage_models.update(original_voyage_models)


@pytest.mark.asyncio
async def test_periodic_reload_reschedules_after_failed_attempt(monkeypatch):
    scheduled_deadlines: list[str] = []
    current_times = iter(
        (
            datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 18, 0, 6, tzinfo=timezone.utc),
        )
    )
    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_REQUIRE_REMOTE", "true")

    async def sleep_until_second_schedule(interval_seconds: float) -> None:
        assert interval_seconds == 300
        next_run_at = get_model_cost_map_source_info()["next_run_at"]
        assert isinstance(next_run_at, str)
        scheduled_deadlines.append(next_run_at)
        if len(scheduled_deadlines) > 1:
            raise asyncio.CancelledError

    def fail_to_load(url: str) -> PreparedModelCostMap:
        raise ModelCostMapLoadError("unavailable")

    manager = ModelCostMapManager(
        load_required_remote_model_cost_map=fail_to_load,
        sleep=sleep_until_second_schedule,
        utc_now=lambda: next(current_times),
    )

    with pytest.raises(asyncio.CancelledError):
        await manager.run_periodic_reload(300)

    assert scheduled_deadlines == [
        "2026-08-18T00:05:00+00:00",
        "2026-08-18T00:11:00+00:00",
    ]
    assert get_model_cost_map_source_info()["next_run_at"] is None


def test_publish_replaces_catalog_entries_and_preserves_runtime_entries():
    original_model_cost = litellm.model_cost
    original_voyage_models = set(litellm.voyage_models)
    original_snapshot = get_required_remote_model_cost_map_snapshot()
    old_catalog_keys = frozenset({"remote-old", "remote-old-alias"})
    litellm.model_cost = {
        "remote-old": {"litellm_provider": "anthropic"},
        "remote-old-alias": {"litellm_provider": "anthropic"},
        "deployment-id": {"input_cost_per_token": 1.0, "litellm_provider": "anthropic"},
    }
    set_required_remote_model_cost_map_keys(old_catalog_keys)
    prepared = PreparedModelCostMap(
        model_cost={"remote-new": {"litellm_provider": "voyage"}},
        catalog_snapshot={"remote-new": {"litellm_provider": "voyage"}},
    )

    try:
        published = ModelCostMapManager._publish(prepared)

        assert published == {
            "deployment-id": {"input_cost_per_token": 1.0, "litellm_provider": "anthropic"},
            "remote-new": {"litellm_provider": "voyage"},
        }
        assert get_required_remote_model_cost_map_keys() == frozenset({"remote-new"})
        assert "remote-new" in litellm.voyage_models
        assert get_required_remote_model_cost_map_snapshot() == {
            "remote-new": {"litellm_provider": "voyage"}
        }
    finally:
        litellm.model_cost = original_model_cost
        set_required_remote_model_cost_map_snapshot(original_snapshot)
        set_required_remote_model_cost_map_keys(frozenset())
        litellm.voyage_models.clear()
        litellm.voyage_models.update(original_voyage_models)
