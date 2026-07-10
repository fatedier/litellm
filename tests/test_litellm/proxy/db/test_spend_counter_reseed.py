import asyncio
from types import SimpleNamespace

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed


class _BlockingUserTable:
    def __init__(self, spend: float, entered: asyncio.Event, release: asyncio.Event):
        self._spend = spend
        self._entered = entered
        self._release = release

    async def find_unique(self, where: dict) -> SimpleNamespace:
        self._entered.set()
        await self._release.wait()
        return SimpleNamespace(spend=self._spend)


def _fake_prisma(user_table: _BlockingUserTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_usertable=user_table))


@pytest.mark.asyncio
async def test_repair_during_cold_reseed_does_not_double_counter():
    """
    Regression: with no Redis, `SpendCounterReseed.coalesced` seeds a cold
    counter via [in-memory check -> await DB read -> increment by db_spend].
    `_repair_stale_spend_counter` (reached from budget-reservation
    reconcile/release when the counter is cold) used to set the counter to
    db_spend without taking the per-counter reseed lock. When that set landed
    inside the reseed's DB-read window, the increment stacked on top of it and
    the counter became exactly 2x the authoritative spend, blocking users with
    spurious BudgetExceededError.
    """
    counter_key = "spend:user:reseed-race-user"
    cache = proxy_server.spend_counter_cache
    cache.in_memory_cache.delete_cache(key=counter_key)

    entered = asyncio.Event()
    release = asyncio.Event()
    prisma = _fake_prisma(_BlockingUserTable(spend=100.0, entered=entered, release=release))

    try:
        reseed_task = asyncio.create_task(
            SpendCounterReseed.coalesced(
                prisma_client=prisma,
                spend_counter_cache=cache,
                counter_key=counter_key,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)

        repair_task = asyncio.create_task(
            proxy_server._repair_stale_spend_counter(counter_key=counter_key, db_spend=100.0)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        release.set()
        assert await asyncio.wait_for(reseed_task, timeout=5) == 100.0
        await asyncio.wait_for(repair_task, timeout=5)

        assert float(cache.in_memory_cache.get_cache(key=counter_key)) == 100.0
    finally:
        cache.in_memory_cache.delete_cache(key=counter_key)


@pytest.mark.asyncio
async def test_repair_stale_spend_counter_only_raises_counter():
    counter_key = "spend:user:reseed-monotonic-user"
    cache = proxy_server.spend_counter_cache
    try:
        cache.in_memory_cache.set_cache(key=counter_key, value=150.0)

        await proxy_server._repair_stale_spend_counter(counter_key=counter_key, db_spend=100.0)
        assert float(cache.in_memory_cache.get_cache(key=counter_key)) == 150.0

        await proxy_server._repair_stale_spend_counter(counter_key=counter_key, db_spend=225.0)
        assert float(cache.in_memory_cache.get_cache(key=counter_key)) == 225.0
    finally:
        cache.in_memory_cache.delete_cache(key=counter_key)
