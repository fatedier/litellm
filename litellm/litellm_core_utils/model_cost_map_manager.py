from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,  # pyright: ignore[reportUnknownVariableType]  # legacy API is untyped
)
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    ModelCostMap,
    ModelCostMapLoadError,
    PreparedModelCostMap,
    get_required_remote_model_cost_map_keys,
    is_remote_model_cost_map_required,
    record_required_remote_model_cost_map_attempt,
    record_required_remote_model_cost_map_failure,
    record_required_remote_model_cost_map_success,
    set_required_remote_model_cost_map_keys,
    set_required_remote_model_cost_map_next_run_at,
    set_required_remote_model_cost_map_snapshot,
)
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,  # pyright: ignore[reportPrivateUsage]  # reuse existing invalidation
)


def _load_model_cost_map(url: str) -> ModelCostMap:
    from litellm.litellm_core_utils.get_model_cost_map import (
        get_model_cost_map,  # pyright: ignore[reportUnknownVariableType]  # legacy API
    )

    return get_model_cost_map(url)  # pyright: ignore[reportUnknownVariableType]  # legacy API


class ModelCostMapLoader(Protocol):
    def __call__(self, url: str) -> ModelCostMap: ...


class RequiredRemoteModelCostMapLoader(Protocol):
    def __call__(self, url: str) -> PreparedModelCostMap: ...


class Sleep(Protocol):
    async def __call__(self, delay: float) -> None: ...


class UtcNow(Protocol):
    def __call__(self) -> datetime: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ModelCostMapManager:
    load_model_cost_map: ModelCostMapLoader = _load_model_cost_map
    load_required_remote_model_cost_map: RequiredRemoteModelCostMapLoader = (
        GetModelCostMap.load_required_remote_model_cost_map
    )
    sleep: Sleep = asyncio.sleep
    utc_now: UtcNow = _utc_now
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reload_model_cost_map(self) -> ModelCostMap:
        async with self.lock:
            if not is_remote_model_cost_map_required():
                model_cost = await asyncio.to_thread(self.load_model_cost_map, litellm.model_cost_map_url)
                return self._replace(model_cost)

            record_required_remote_model_cost_map_attempt()
            try:
                prepared = await asyncio.to_thread(
                    self.load_required_remote_model_cost_map,
                    litellm.model_cost_map_url,
                )
                return self._publish(prepared)
            except ModelCostMapLoadError as exc:
                record_required_remote_model_cost_map_failure(exc)
                raise

    @staticmethod
    def _publish(prepared: PreparedModelCostMap) -> ModelCostMap:
        """Replace the authoritative catalog while preserving runtime registrations."""
        previous_catalog_keys = get_required_remote_model_cost_map_keys()
        runtime_model_cost: ModelCostMap = litellm.model_cost  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy global
        next_model_cost = {**runtime_model_cost}  # mutable-ok: runtime registry snapshot
        for model_name in previous_catalog_keys:
            next_model_cost.pop(model_name, None)
        next_model_cost.update(prepared.model_cost)
        set_fallback_generalizations(None)
        litellm.model_cost = next_model_cost
        _invalidate_model_cost_lowercase_map()
        # Required Proxy calls resolve model groups to provider-qualified Router upstream models.
        # Keep LiteLLM's existing incremental registry update instead of rebuilding global registries.
        litellm.add_known_models(model_cost_map=prepared.model_cost)  # pyright: ignore[reportUnknownMemberType]  # legacy API is untyped
        catalog_keys = frozenset(prepared.model_cost)
        set_required_remote_model_cost_map_snapshot(prepared.catalog_snapshot)
        set_required_remote_model_cost_map_keys(catalog_keys)
        record_required_remote_model_cost_map_success(len(prepared.catalog_snapshot))
        return next_model_cost

    @staticmethod
    def _replace(model_cost: ModelCostMap) -> ModelCostMap:
        litellm.model_cost = model_cost
        _invalidate_model_cost_lowercase_map()
        litellm.add_known_models(model_cost_map=model_cost)  # pyright: ignore[reportUnknownMemberType]  # legacy API is untyped
        set_required_remote_model_cost_map_snapshot(None)
        set_required_remote_model_cost_map_keys(frozenset())
        return model_cost

    async def run_periodic_reload(self, interval_seconds: int) -> None:
        try:
            while True:
                next_run_at = self.utc_now() + timedelta(seconds=interval_seconds)
                set_required_remote_model_cost_map_next_run_at(next_run_at.isoformat())
                await self.sleep(interval_seconds)
                set_required_remote_model_cost_map_next_run_at(None)
                try:
                    await self.reload_model_cost_map()
                except ModelCostMapLoadError as exc:
                    verbose_logger.error(
                        "LiteLLM: Required remote model cost map reload failed; retaining last successful map: %s",
                        exc,
                    )
        finally:
            set_required_remote_model_cost_map_next_run_at(None)


model_cost_map_manager = ModelCostMapManager()
