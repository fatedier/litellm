from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, List, Optional, Protocol, Set, runtime_checkable
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_AccessGroupTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _cache_access_object,
    _cache_key_object,
    _cache_team_object,
    _get_team_object_from_cache,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.management_helpers.access_group_team_sync import invalidate_access_group_cache
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.repositories.table_repositories import AccessGroupRepository
from litellm.types.access_group import (
    AccessGroupCreateRequest,
    AccessGroupResponse,
    AccessGroupUpdateRequest,
)

router: Final = APIRouter(
    tags=["access group management"],
)


class _AccessGroupRecord(Protocol):
    @property
    def access_group_id(self) -> str: ...

    @property
    def assigned_team_ids(self) -> Sequence[str] | None: ...

    @property
    def assigned_key_ids(self) -> Sequence[str] | None: ...

    def dict(self) -> Mapping[str, object]: ...


class _TeamRecord(Protocol):
    @property
    def team_id(self) -> str: ...

    @property
    def access_group_ids(self) -> Sequence[str] | None: ...


class _KeyRecord(Protocol):
    @property
    def token(self) -> str: ...

    @property
    def access_group_ids(self) -> Sequence[str] | None: ...


class _AccessGroupTable(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> _AccessGroupRecord | None: ...

    async def find_many(self, order: Mapping[str, object]) -> Sequence[_AccessGroupRecord]: ...

    async def create(self, data: Mapping[str, object]) -> _AccessGroupRecord: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> _AccessGroupRecord: ...

    async def delete(self, where: Mapping[str, object]) -> object: ...


class _TeamTable(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> _TeamRecord | None: ...

    async def find_many(self, where: Mapping[str, object]) -> Sequence[_TeamRecord]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> object: ...


class _KeyTable(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> _KeyRecord | None: ...

    async def find_many(self, where: Mapping[str, object]) -> Sequence[_KeyRecord]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> object: ...


class _AccessGroupTx(Protocol):
    @property
    def litellm_accessgrouptable(self) -> _AccessGroupTable: ...

    @property
    def litellm_teamtable(self) -> _TeamTable: ...

    @property
    def litellm_verificationtoken(self) -> _KeyTable: ...


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


def _require_admin_view(user_api_key_dict: UserAPIKeyAuth) -> None:
    """Admin Viewer parity: PROXY_ADMIN or PROXY_ADMIN_VIEW_ONLY may read."""
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


@runtime_checkable
class _MembershipCountTable(Protocol):
    """The two reads the member count needs, as a typed boundary.

    The repository exposes its table as ``Any``, so calling through it directly
    would spread that through every count. Narrowing once here keeps the rest
    typed without annotating the Prisma client itself.
    """

    async def count(self, where: Mapping[str, object]) -> int: ...

    async def group_by(
        self,
        by: Sequence[str],
        where: Mapping[str, object] | None = ...,
        count: Mapping[str, bool] | None = ...,
    ) -> object: ...


class _MemberCount(BaseModel):
    user_id: int


class _GroupMemberCountRow(BaseModel):
    access_group_id: str
    member_count: _MemberCount = Field(alias="_count")


_GROUP_MEMBER_COUNT_ROWS = TypeAdapter(tuple[_GroupMemberCountRow, ...])


def _require_membership_table(prisma_client: object) -> _MembershipCountTable:
    table = getattr(getattr(prisma_client, "db", None), "litellm_useraccessgroupmembership", None)
    if not isinstance(table, _MembershipCountTable):
        raise TypeError("database does not expose the access group membership table")
    return table


async def _count_group_members(prisma_client: object, access_group_id: str) -> int:
    """Count the users assigned to one group.

    Counts the join table alone. Teams and keys can also be assigned to a group,
    but those are already on the response as ``assigned_team_ids`` and
    ``assigned_key_ids``; folding them in would make the number mean nothing in
    particular.
    """
    return await _require_membership_table(prisma_client).count(where={"access_group_id": access_group_id})


async def _count_members_by_group(
    prisma_client: object, access_group_ids: Optional[Sequence[str]] = None
) -> Mapping[str, int]:
    """Count members for many groups in one aggregate.

    Asking per group would issue one query per row of the listing. A group
    nobody belongs to produces no row here, so callers must default it to zero
    rather than indexing the result.
    """
    rows = await _require_membership_table(prisma_client).group_by(
        by=["access_group_id"],
        where={"access_group_id": {"in": list(access_group_ids)}} if access_group_ids is not None else None,
        count={"user_id": True},
    )
    return MappingProxyType(
        {row.access_group_id: row.member_count.user_id for row in _GROUP_MEMBER_COUNT_ROWS.validate_python(rows)}
    )


def _record_to_response(record, *, user_count: int) -> AccessGroupResponse:
    return AccessGroupResponse(
        access_group_id=record.access_group_id,
        access_group_name=record.access_group_name,
        description=record.description,
        access_model_names=record.access_model_names,
        access_mcp_server_ids=record.access_mcp_server_ids,
        access_agent_ids=record.access_agent_ids,
        assigned_team_ids=record.assigned_team_ids,
        assigned_key_ids=record.assigned_key_ids,
        created_at=record.created_at,
        created_by=record.created_by,
        updated_at=record.updated_at,
        updated_by=record.updated_by,
        user_count=user_count,
    )


def _record_to_access_group_table(record: _AccessGroupRecord) -> LiteLLM_AccessGroupTable:
    """Convert a Prisma record to a LiteLLM_AccessGroupTable pydantic object for caching."""
    return LiteLLM_AccessGroupTable.model_validate(record.dict())


async def _cache_access_group_record(record: _AccessGroupRecord) -> None:
    """
    Cache an access group Prisma record in the user_api_key_cache.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    access_group_table: Final = _record_to_access_group_table(record)
    await _cache_access_object(
        access_group_id=record.access_group_id,
        access_group_table=access_group_table,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


# ---------------------------------------------------------------------------
# DB sync helpers (called inside a Prisma transaction)
# ---------------------------------------------------------------------------


async def _sync_add_access_group_to_teams(tx: _AccessGroupTx, team_ids: list[str], access_group_id: str) -> None:
    """Add access_group_id to each team's access_group_ids (idempotent)."""
    for team_id in team_ids:
        team = await tx.litellm_teamtable.find_unique(where={"team_id": team_id})
        if team is not None and access_group_id not in (team.access_group_ids or []):
            await tx.litellm_teamtable.update(
                where={"team_id": team_id},
                data={"access_group_ids": list(team.access_group_ids or []) + [access_group_id]},
            )


async def _sync_remove_access_group_from_teams(tx: _AccessGroupTx, team_ids: list[str], access_group_id: str) -> None:
    """Remove access_group_id from each team's access_group_ids (idempotent)."""
    for team_id in team_ids:
        team = await tx.litellm_teamtable.find_unique(where={"team_id": team_id})
        if team is not None and access_group_id in (team.access_group_ids or []):
            await tx.litellm_teamtable.update(
                where={"team_id": team_id},
                data={"access_group_ids": [ag for ag in (team.access_group_ids or ()) if ag != access_group_id]},
            )


async def _sync_add_access_group_to_keys(tx: _AccessGroupTx, key_tokens: list[str], access_group_id: str) -> None:
    """Add access_group_id to each key's access_group_ids (idempotent)."""
    for token in key_tokens:
        key = await tx.litellm_verificationtoken.find_unique(where={"token": token})
        if key is not None and access_group_id not in (key.access_group_ids or []):
            await tx.litellm_verificationtoken.update(
                where={"token": token},
                data={"access_group_ids": list(key.access_group_ids or []) + [access_group_id]},
            )


async def _sync_remove_access_group_from_keys(tx: _AccessGroupTx, key_tokens: list[str], access_group_id: str) -> None:
    """Remove access_group_id from each key's access_group_ids (idempotent)."""
    for token in key_tokens:
        key = await tx.litellm_verificationtoken.find_unique(where={"token": token})
        if key is not None and access_group_id in (key.access_group_ids or []):
            await tx.litellm_verificationtoken.update(
                where={"token": token},
                data={"access_group_ids": [ag for ag in (key.access_group_ids or ()) if ag != access_group_id]},
            )


# ---------------------------------------------------------------------------
# Cache patch helpers
# ---------------------------------------------------------------------------


async def _patch_team_caches_add_access_group(
    team_ids: list[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached team objects to include access_group_id."""
    for team_id in team_ids:
        cached_team = await _get_team_object_from_cache(
            key=f"team_id:{team_id}",
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
        )
        if cached_team is None:
            continue
        if cached_team.access_group_ids is None:
            cached_team.access_group_ids = [access_group_id]
        elif access_group_id not in cached_team.access_group_ids:
            cached_team.access_group_ids = list(cached_team.access_group_ids) + [access_group_id]
        else:
            continue
        await _cache_team_object(
            team_id=team_id,
            team_table=cached_team,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


async def _patch_team_caches_remove_access_group(
    team_ids: list[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached team objects to remove access_group_id."""
    for team_id in team_ids:
        cached_team = await _get_team_object_from_cache(
            key=f"team_id:{team_id}",
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
        )
        if cached_team is not None and cached_team.access_group_ids:
            cached_team.access_group_ids = [ag for ag in cached_team.access_group_ids if ag != access_group_id]
            await _cache_team_object(
                team_id=team_id,
                team_table=cached_team,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )


async def _patch_key_caches_add_access_group(
    key_tokens: list[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached key objects to include access_group_id."""
    for token in key_tokens:
        cached_key = await user_api_key_cache.async_get_cache(
            key=token,
            model_type=UserAPIKeyAuth,
        )
        if cached_key is None:
            continue
        if cached_key.access_group_ids is None:
            cached_key.access_group_ids = [access_group_id]
        elif access_group_id not in cached_key.access_group_ids:
            cached_key.access_group_ids = list(cached_key.access_group_ids) + [access_group_id]
        else:
            continue
        await _cache_key_object(
            hashed_token=token,
            user_api_key_obj=cached_key,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


async def _patch_key_caches_remove_access_group(
    key_tokens: list[str],
    access_group_id: str,
    user_api_key_cache,
    proxy_logging_obj,
) -> None:
    """Patch cached key objects to remove access_group_id."""
    for token in key_tokens:
        cached_key = await user_api_key_cache.async_get_cache(
            key=token,
            model_type=UserAPIKeyAuth,
        )
        if cached_key is not None and cached_key.access_group_ids:
            cached_key.access_group_ids = [ag for ag in cached_key.access_group_ids if ag != access_group_id]
            await _cache_key_object(
                hashed_token=token,
                user_api_key_obj=cached_key,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v1/access_group",
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_access_group(
    data: AccessGroupCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    try:
        tx: _AccessGroupTx
        async with prisma_client.db.tx() as tx:
            existing: Final = await tx.litellm_accessgrouptable.find_unique(
                where={"access_group_name": data.access_group_name}
            )
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Access group '{data.access_group_name}' already exists",
                )

            record: Final = await tx.litellm_accessgrouptable.create(
                data={
                    "access_group_name": data.access_group_name,
                    "description": data.description,
                    "access_model_names": data.access_model_names or [],
                    "access_mcp_server_ids": data.access_mcp_server_ids or [],
                    "access_agent_ids": data.access_agent_ids or [],
                    "assigned_team_ids": data.assigned_team_ids or [],
                    "assigned_key_ids": data.assigned_key_ids or [],
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                }
            )

            # Sync team and key tables to reference the new access group
            await _sync_add_access_group_to_teams(tx, data.assigned_team_ids or [], record.access_group_id)
            await _sync_add_access_group_to_keys(tx, data.assigned_key_ids or [], record.access_group_id)
    except HTTPException:
        raise
    except Exception as e:
        # Race condition: another request created the same name between find_unique and create.
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{data.access_group_name}' already exists",
            )
        raise

    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _cache_access_group_record(record)
    await _patch_team_caches_add_access_group(
        data.assigned_team_ids or [],
        record.access_group_id,
        user_api_key_cache,
        proxy_logging_obj,
    )
    await _patch_key_caches_add_access_group(
        data.assigned_key_ids or [],
        record.access_group_id,
        user_api_key_cache,
        proxy_logging_obj,
    )

    return _record_to_response(record, user_count=0)


@router.get(
    "/v1/access_group",
    response_model=list[AccessGroupResponse],
)
async def list_access_groups(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list[AccessGroupResponse]:
    _require_admin_view(user_api_key_dict)
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    table: Final[_AccessGroupTable] = AccessGroupRepository(prisma_client).table
    records: Final = await table.find_many(order={"created_at": "desc"})
    member_counts = await _count_members_by_group(prisma_client)
    return [_record_to_response(r, user_count=member_counts.get(r.access_group_id, 0)) for r in records]


@router.get(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def get_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_admin_view(user_api_key_dict)
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    table: Final[_AccessGroupTable] = AccessGroupRepository(prisma_client).table
    record: Final = await table.find_unique(where={"access_group_id": access_group_id})
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )
    return _record_to_response(record, user_count=await _count_group_members(prisma_client, access_group_id))


@router.put(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def update_access_group(
    access_group_id: str,
    data: AccessGroupUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    update_fields: Final = data.model_dump(exclude_unset=True)
    update_data: Final[dict] = {"updated_by": user_api_key_dict.user_id}
    for field, value in update_fields.items():
        if (
            field
            in (
                "assigned_team_ids",
                "assigned_key_ids",
                "access_model_names",
                "access_mcp_server_ids",
                "access_agent_ids",
            )
            and value is None
        ):
            value = []
        update_data[field] = value

    # Initialize delta lists before the try block so they remain accessible
    # for cache updates after the transaction, even if an error path is added later.
    teams_to_add: list[str] = []
    teams_to_remove: list[str] = []
    keys_to_add: list[str] = []
    keys_to_remove: list[str] = []

    try:
        tx: _AccessGroupTx
        async with prisma_client.db.tx() as tx:
            # Read inside the transaction so delta computation is consistent with the write,
            # avoiding a TOCTOU race where a concurrent update could make deltas stale.
            existing: Final = await tx.litellm_accessgrouptable.find_unique(where={"access_group_id": access_group_id})
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Access group '{access_group_id}' not found",
                )

            old_team_ids: Final[set[str]] = set(existing.assigned_team_ids or [])
            old_key_ids: Final[set[str]] = set(existing.assigned_key_ids or [])
            new_team_ids: Final[set[str]] = (
                set(update_fields["assigned_team_ids"] or []) if "assigned_team_ids" in update_fields else old_team_ids
            )
            new_key_ids: Final[set[str]] = (
                set(update_fields["assigned_key_ids"] or []) if "assigned_key_ids" in update_fields else old_key_ids
            )

            teams_to_add = list(new_team_ids - old_team_ids)
            teams_to_remove = list(old_team_ids - new_team_ids)
            keys_to_add = list(new_key_ids - old_key_ids)
            keys_to_remove = list(old_key_ids - new_key_ids)

            record: Final = await tx.litellm_accessgrouptable.update(
                where={"access_group_id": access_group_id},
                data=update_data,
            )

            await _sync_add_access_group_to_teams(tx, teams_to_add, access_group_id)
            await _sync_remove_access_group_from_teams(tx, teams_to_remove, access_group_id)
            await _sync_add_access_group_to_keys(tx, keys_to_add, access_group_id)
            await _sync_remove_access_group_from_keys(tx, keys_to_remove, access_group_id)
    except HTTPException:
        raise
    except Exception as e:
        # Unique constraint violation (e.g. access_group_name already exists).
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{update_data.get('access_group_name', '')}' already exists",
            )
        raise

    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _cache_access_group_record(record)
    await _patch_team_caches_add_access_group(teams_to_add, access_group_id, user_api_key_cache, proxy_logging_obj)
    await _patch_team_caches_remove_access_group(
        teams_to_remove, access_group_id, user_api_key_cache, proxy_logging_obj
    )
    await _patch_key_caches_add_access_group(keys_to_add, access_group_id, user_api_key_cache, proxy_logging_obj)
    await _patch_key_caches_remove_access_group(keys_to_remove, access_group_id, user_api_key_cache, proxy_logging_obj)

    return _record_to_response(record, user_count=await _count_group_members(prisma_client, access_group_id))


@router.delete(
    "/v1/access_group/{access_group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> None:
    _require_proxy_admin(user_api_key_dict)
    prisma_client: Final = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    try:
        affected_team_ids: list[str] = []
        affected_key_tokens: list[str] = []

        tx: _AccessGroupTx
        async with prisma_client.db.tx() as tx:
            existing: Final = await tx.litellm_accessgrouptable.find_unique(where={"access_group_id": access_group_id})
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Access group '{access_group_id}' not found",
                )

            assigned_user_count = await tx.litellm_useraccessgroupmembership.count(
                where={"access_group_id": access_group_id}
            )
            if assigned_user_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Access group '{access_group_id}' is still assigned to {assigned_user_count} user(s). "
                        "Clear those users' access_group_ids before deleting it."
                    ),
                )

            # Union of: teams that have this access_group_id in their own access_group_ids
            # AND teams listed in assigned_team_ids (handles out-of-sync data from before this sync was added)
            teams_with_group: Final = await tx.litellm_teamtable.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            all_affected_team_ids: Final[set[str]] = {team.team_id for team in teams_with_group} | set(
                existing.assigned_team_ids or []
            )
            affected_team_ids = list(all_affected_team_ids)

            # Union of: keys that have this access_group_id in their own access_group_ids
            # AND keys listed in assigned_key_ids (handles out-of-sync data)
            keys_with_group: Final = await tx.litellm_verificationtoken.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            all_affected_key_tokens: Final[set[str]] = {key.token for key in keys_with_group} | set(
                existing.assigned_key_ids or []
            )
            affected_key_tokens = list(all_affected_key_tokens)

            # Update teams returned by find_many directly — we already have their data.
            for team in teams_with_group:
                await tx.litellm_teamtable.update(
                    where={"team_id": team.team_id},
                    data={"access_group_ids": [ag for ag in (team.access_group_ids or ()) if ag != access_group_id]},
                )
            # Use _sync_remove only for out-of-sync teams not found by the hasSome query.
            out_of_sync_team_ids: Final = set(existing.assigned_team_ids or []) - {t.team_id for t in teams_with_group}
            await _sync_remove_access_group_from_teams(tx, list(out_of_sync_team_ids), access_group_id)

            # Update keys returned by find_many directly — we already have their data.
            for key in keys_with_group:
                await tx.litellm_verificationtoken.update(
                    where={"token": key.token},
                    data={"access_group_ids": [ag for ag in (key.access_group_ids or ()) if ag != access_group_id]},
                )
            # Use _sync_remove only for out-of-sync keys not found by the hasSome query.
            out_of_sync_key_tokens: Final = set(existing.assigned_key_ids or []) - {k.token for k in keys_with_group}
            await _sync_remove_access_group_from_keys(tx, list(out_of_sync_key_tokens), access_group_id)

            await tx.litellm_accessgrouptable.delete(where={"access_group_id": access_group_id})

        from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

        await invalidate_access_group_cache(access_group_id)
        await _patch_team_caches_remove_access_group(
            affected_team_ids, access_group_id, user_api_key_cache, proxy_logging_obj
        )
        await _patch_key_caches_remove_access_group(
            affected_key_tokens, access_group_id, user_api_key_cache, proxy_logging_obj
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "delete_access_group failed: access_group_id=%s error=%s",
            access_group_id,
            e,
        )
        if PrismaDBExceptionHandler.is_database_infrastructure_error(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        if "P2025" in str(e) or ("record" in str(e).lower() and "not found" in str(e).lower()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Access group '{access_group_id}' not found",
            )
        if "P2003" in str(e) or "foreign key constraint" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Access group '{access_group_id}' is still assigned to one or more users. "
                    "Clear those users' access_group_ids before deleting it."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete access group. Please try again.",
        )


# Alias routes for /v1/unified_access_group
router.add_api_route(
    "/v1/unified_access_group",
    create_access_group,
    methods=["POST"],
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
router.add_api_route(
    "/v1/unified_access_group",
    list_access_groups,
    methods=["GET"],
    response_model=list[AccessGroupResponse],
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    get_access_group,
    methods=["GET"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    update_access_group,
    methods=["PUT"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    delete_access_group,
    methods=["DELETE"],
    status_code=status.HTTP_204_NO_CONTENT,
)
