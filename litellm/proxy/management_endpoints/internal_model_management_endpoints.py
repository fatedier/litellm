"""Internal model deployment updates with exact field-presence semantics."""

import asyncio
import copy
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Protocol, cast, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, Response, status
from prisma import Json
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_ProxyModelTable,
    LitellmTableNames,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    user_api_key_has_admin_view,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils import openai_endpoint_utils
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.types.router import ModelInfo, SPECIAL_MODEL_INFO_PARAMS, updateLiteLLMParams
from litellm.utils import get_utc_datetime

router = APIRouter()

_MAX_PATCH_ATTEMPTS = 3
_ALLOWED_MODEL_INFO_FIELDS = frozenset(("base_model",))
_LITELLM_PARAMS_TOMBSTONE_DENYLIST = frozenset(("model",))
_SPECIAL_MODEL_INFO_FIELDS = frozenset(SPECIAL_MODEL_INFO_PARAMS)
_EMPTY_FIELD_SET: frozenset[str] = frozenset()
_JSON_OBJECT_ADAPTER = TypeAdapter(Mapping[str, JsonValue])
_JSON_OBJECT_DICT_ADAPTER = TypeAdapter(dict[str, JsonValue])
_EMPTY_JSON_OBJECT: Mapping[str, JsonValue] = MappingProxyType({})  # mutable-ok: wrapped in a read-only proxy


@dataclass(frozen=True, slots=True)
class _JsonObjectWithoutKeys(Mapping[str, JsonValue]):
    source: Mapping[str, JsonValue]
    excluded_keys: frozenset[str]

    def __getitem__(self, key: str) -> JsonValue:
        if key in self.excluded_keys:
            raise KeyError(key)
        return self.source[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.source if key not in self.excluded_keys)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _StoredModel(Protocol):
    model_id: str
    model_name: str
    litellm_params: object
    model_info: object
    blocked: bool
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None

    def dict(self) -> Mapping[str, object]: ...


@runtime_checkable
class _ModelDetailTable(Protocol):
    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> _StoredModel | None: ...

    async def update_many(self, data: Mapping[str, object], where: Mapping[str, object]) -> int: ...


@runtime_checkable
class _ModelDeleteTable(Protocol):
    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> _StoredModel | None: ...

    async def delete_many(self, where: Mapping[str, object]) -> int: ...


@runtime_checkable
class _ModelTable(_ModelDetailTable, Protocol):
    async def find_many(
        self,
        take: int | None = None,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, str] | list[Mapping[str, str]] | None = None,
        distinct: list[str] | None = None,
    ) -> list[_StoredModel]: ...


@runtime_checkable
class _DeploymentSanitizer(Protocol):
    def __call__(
        self,
        deployment_dict: Mapping[str, object],
        excluded_keys: frozenset[str] | None = None,
    ) -> object: ...


class _RouterDeploymentParams(Protocol):
    model: str | None


class _RouterDeployment(Protocol):
    model_name: str | None
    litellm_params: _RouterDeploymentParams


def _require_deployment_sanitizer() -> _DeploymentSanitizer:
    candidate: object = getattr(openai_endpoint_utils, "remove_sensitive_info_from_deployment", None)
    if not isinstance(candidate, _DeploymentSanitizer):
        raise TypeError("deployment sanitizer is unavailable")
    return candidate


def _require_model_table(writer_db: object) -> _ModelDetailTable:
    model_table = getattr(writer_db, "litellm_proxymodeltable", None)
    if not isinstance(model_table, _ModelDetailTable):
        raise TypeError("writer database does not expose the model deployment table")
    return model_table


def _require_model_list_table(writer_db: object) -> _ModelTable:
    model_table = getattr(writer_db, "litellm_proxymodeltable", None)
    if not isinstance(model_table, _ModelTable):
        raise TypeError("writer database does not expose the model deployment list table")
    return model_table


def _require_model_delete_table(writer_db: object) -> _ModelDeleteTable:
    model_table = getattr(writer_db, "litellm_proxymodeltable", None)
    if not isinstance(model_table, _ModelDeleteTable):
        raise TypeError("writer database does not expose the model deployment delete table")
    return model_table


def _router_deployment_name_and_model(deployment: object) -> tuple[str | None, str | None]:
    if deployment is None:
        return None, None
    if isinstance(deployment, Mapping):
        deployment_mapping = cast(Mapping[str, object], deployment)
        deployment_name = deployment_mapping.get("model_name")
        params = deployment_mapping.get("litellm_params")
        model = cast(Mapping[str, object], params).get("model") if isinstance(params, Mapping) else None
        return (
            deployment_name if isinstance(deployment_name, str) else None,
            model if isinstance(model, str) else None,
        )
    router_deployment = cast(_RouterDeployment, deployment)
    return router_deployment.model_name, router_deployment.litellm_params.model


def _split_litellm_params_patch(
    value: Mapping[str, JsonValue],
) -> tuple[Mapping[str, JsonValue], frozenset[str]]:
    tombstones = frozenset(field for field, field_value in value.items() if field_value is None)
    denied_fields = tuple(sorted(tombstones & _LITELLM_PARAMS_TOMBSTONE_DENYLIST))
    if denied_fields:
        raise ValueError(f"removing litellm_params fields is not supported: {', '.join(denied_fields)}")

    patch_values = _JsonObjectWithoutKeys(value, tombstones)
    if patch_values:
        model = patch_values.get("model")
        if isinstance(model, str) and not model.strip():
            raise ValueError("model must not be empty")
        updateLiteLLMParams.model_validate(copy.deepcopy(_JSON_OBJECT_DICT_ADAPTER.validate_python(patch_values)))
    return patch_values, tombstones


def _split_model_info_patch(
    value: Mapping[str, JsonValue],
) -> tuple[Mapping[str, JsonValue], frozenset[str]]:
    unsupported_fields = tuple(sorted(frozenset(value) - _ALLOWED_MODEL_INFO_FIELDS))
    if unsupported_fields:
        raise ValueError(f"updating model_info fields is not supported: {', '.join(unsupported_fields)}")

    base_model = value.get("base_model")
    if isinstance(base_model, str) and not base_model.strip():
        raise ValueError("base_model must not be empty")

    tombstones = frozenset(field for field, field_value in value.items() if field_value is None)
    patch_values = _JsonObjectWithoutKeys(value, tombstones)
    if patch_values:
        ModelInfo.model_validate(copy.deepcopy(_JSON_OBJECT_DICT_ADAPTER.validate_python(patch_values)))
    return patch_values, tombstones


class InternalModelDeploymentPatch(BaseModel):
    model_name: str | None = None
    litellm_params: Mapping[str, JsonValue] | None = None
    model_info: Mapping[str, JsonValue] | None = None
    blocked: bool | None = None

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model_name must not be empty")
        return value

    @field_validator("litellm_params")
    @classmethod
    def validate_litellm_params_patch(
        cls, value: Mapping[str, JsonValue] | None, info: ValidationInfo
    ) -> Mapping[str, JsonValue] | None:
        if value is None:
            return value
        if not value:
            raise ValueError(f"{info.field_name} must contain at least one field")
        _split_litellm_params_patch(value)
        return value

    @field_validator("model_info")
    @classmethod
    def validate_model_info_patch(
        cls, value: Mapping[str, JsonValue] | None, info: ValidationInfo
    ) -> Mapping[str, JsonValue] | None:
        if value is None:
            return value
        if not value:
            raise ValueError(f"{info.field_name} must contain at least one field")
        _split_model_info_patch(value)
        return value

    @model_validator(mode="after")
    def validate_patch(self) -> "InternalModelDeploymentPatch":
        if not self.model_fields_set:
            raise ValueError("patch must contain at least one field")
        null_fields = sorted(field for field in self.model_fields_set if getattr(self, field) is None)
        if null_fields:
            raise ValueError(f"removing model deployment fields is not supported: {', '.join(null_fields)}")
        return self


class InternalModelDeploymentPatchResponse(BaseModel):
    model_id: str


class InternalModelDeploymentDeleteResponse(BaseModel):
    model_id: str


class InternalModelDeploymentResponse(BaseModel):
    model_id: str
    model_name: str
    litellm_params: Mapping[str, JsonValue]
    model_info: Mapping[str, JsonValue] | None
    blocked: bool
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None


class InternalModelDeploymentListResponse(BaseModel):
    data: list[InternalModelDeploymentResponse]


def _json_object(value: object, field_name: str) -> Mapping[str, JsonValue]:
    if value is None:
        return _EMPTY_JSON_OBJECT
    try:
        if isinstance(value, str):
            return _JSON_OBJECT_ADAPTER.validate_json(value)
        return _JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise TypeError(f"stored {field_name} must be a JSON object") from exc


def _decrypt_json_object(value: object, field_name: str) -> Mapping[str, JsonValue]:
    return dict(  # mutable-ok: the sensitive-data helper requires a concrete JSON object
        (
            key,
            decrypt_value_helper(
                field_value,
                key=key,
                exception_type="debug",
                return_original_value=True,
            )
            if isinstance(field_value, str)
            else field_value,
        )
        for key, field_value in _json_object(value, field_name).items()
    )


def _safe_litellm_params(value: object) -> Mapping[str, JsonValue]:
    sanitized_deployment = _JSON_OBJECT_ADAPTER.validate_python(
        _require_deployment_sanitizer()(
            deployment_dict={  # mutable-ok: the existing sanitizer mutates this isolated response object
                "litellm_params": dict(  # mutable-ok: the existing sanitizer requires a concrete dict
                    _decrypt_json_object(value, "litellm_params")
                )
            },
            excluded_keys=frozenset(("litellm_credential_name",)),
        )
    )
    return _json_object(sanitized_deployment.get("litellm_params"), "litellm_params")


def _build_internal_model_deployment_response(
    model_id: str,
    stored_model: _StoredModel,
) -> InternalModelDeploymentResponse:
    return InternalModelDeploymentResponse(
        model_id=model_id,
        model_name=stored_model.model_name,
        litellm_params=_safe_litellm_params(stored_model.litellm_params),
        # model_info is user-defined metadata and is intentionally returned as stored.
        # LiteLLM-managed credentials belong in litellm_params, which is sanitized above.
        model_info=(None if stored_model.model_info is None else _json_object(stored_model.model_info, "model_info")),
        blocked=stored_model.blocked,
        created_at=stored_model.created_at,
        created_by=stored_model.created_by,
        updated_at=stored_model.updated_at,
        updated_by=stored_model.updated_by,
    )


def _next_updated_at(existing_updated_at: datetime | None) -> datetime:
    now = get_utc_datetime()
    if existing_updated_at is None:
        return now
    return max(now, existing_updated_at + timedelta(milliseconds=1))


def _merge_json_object(
    existing_value: object,
    field_name: str,
    patch: Mapping[str, JsonValue],
    tombstones: frozenset[str] = _EMPTY_FIELD_SET,
    *,
    encrypt_strings: bool = False,
) -> Mapping[str, JsonValue]:
    return dict(  # mutable-ok: JSON serialization requires a concrete object after functional merge
        (key, value)
        for key, value in chain(
            _json_object(existing_value, field_name).items(),
            (
                (key, encrypt_value_helper(value) if encrypt_strings and isinstance(value, str) else value)
                for key, value in patch.items()
            ),
        )
        if key not in tombstones
    )


def _validate_merged_litellm_model(
    existing_value: object,
    patch: Mapping[str, JsonValue],
    tombstones: frozenset[str],
) -> None:
    merged = _merge_json_object(
        existing_value,
        "litellm_params",
        patch,
        tombstones,
    )
    model = merged.get("model")
    if isinstance(model, str):
        model = decrypt_value_helper(
            model,
            key="model",
            exception_type="debug",
            return_original_value=True,
        )
    if not isinstance(model, str) or not model.strip():
        raise TypeError("stored litellm_params.model must be a non-empty string")


def _json_update_items(
    existing_value: object,
    field_name: str,
    patch: Mapping[str, JsonValue],
    tombstones: frozenset[str],
    *,
    encrypt_strings: bool = False,
) -> tuple[tuple[str, object], ...]:
    merged = _merge_json_object(
        existing_value,
        field_name,
        patch,
        tombstones,
        encrypt_strings=encrypt_strings,
    )
    return ((field_name, json.dumps(merged)),)


def _build_update_data(
    existing_model: _StoredModel,
    patch: InternalModelDeploymentPatch,
    updated_by: str,
) -> Mapping[str, object]:
    base_items: tuple[tuple[str, object], ...] = (
        ("updated_by", updated_by),
        ("updated_at", _next_updated_at(getattr(existing_model, "updated_at", None))),
    )
    model_name_items: tuple[tuple[str, object], ...] = (
        (("model_name", patch.model_name),) if "model_name" in patch.model_fields_set else ()
    )
    litellm_patch_values, litellm_tombstones = (
        _split_litellm_params_patch(patch.litellm_params)
        if patch.litellm_params is not None
        else (_EMPTY_JSON_OBJECT, _EMPTY_FIELD_SET)
    )
    model_info_patch_values, model_info_tombstones = (
        _split_model_info_patch(patch.model_info)
        if patch.model_info is not None
        else (_EMPTY_JSON_OBJECT, _EMPTY_FIELD_SET)
    )
    pricing_tombstones = litellm_tombstones & _SPECIAL_MODEL_INFO_FIELDS
    effective_model_info_tombstones = model_info_tombstones | pricing_tombstones
    should_update_model_info = patch.model_info is not None or (
        bool(pricing_tombstones) and existing_model.model_info is not None
    )
    _validate_merged_litellm_model(
        existing_model.litellm_params,
        litellm_patch_values,
        litellm_tombstones,
    )
    litellm_params_items = (
        _json_update_items(
            existing_model.litellm_params,
            "litellm_params",
            litellm_patch_values,
            litellm_tombstones,
            encrypt_strings=True,
        )
        if patch.litellm_params is not None
        else ()
    )
    model_info_items = (
        _json_update_items(
            existing_model.model_info,
            "model_info",
            model_info_patch_values,
            effective_model_info_tombstones,
        )
        if should_update_model_info
        else ()
    )
    blocked_items: tuple[tuple[str, object], ...] = (
        (("blocked", patch.blocked),) if "blocked" in patch.model_fields_set else ()
    )
    return dict(  # mutable-ok: Prisma update_many requires a concrete data object
        chain(base_items, model_name_items, litellm_params_items, model_info_items, blocked_items)
    )


def _model_json(model: object) -> str | None:
    if isinstance(model, BaseModel):
        return model.model_dump_json(exclude_none=True)
    model_dump_json = getattr(model, "model_dump_json", None)
    if callable(model_dump_json):
        result = model_dump_json(exclude_none=True)
        return result if isinstance(result, str) else None
    return None


def _build_after_model(existing_model: _StoredModel, update_data: Mapping[str, object]) -> LiteLLM_ProxyModelTable:
    """Build the snapshot represented by this request without rereading the row."""
    if isinstance(existing_model, BaseModel):
        existing_data = existing_model.model_dump()
    else:
        existing_data = existing_model.dict()
    existing_snapshot = LiteLLM_ProxyModelTable.model_validate(existing_data)

    json_fields = frozenset(("litellm_params", "model_info"))
    model_update = dict(  # mutable-ok: Pydantic model_copy requires a concrete update object
        (
            field,
            _json_object(value, field) if field in json_fields else value,
        )
        for field, value in update_data.items()
    )

    return existing_snapshot.model_copy(update=model_update)


def _prisma_json_equals(value: object, field_name: str) -> Mapping[str, object]:
    return {  # mutable-ok: Prisma JSON equality filters require this object shape
        "equals": Json(dict(_json_object(value, field_name)))  # mutable-ok: Prisma Json requires a concrete dict
    }


def _model_id_where(model_id: str) -> Mapping[str, object]:
    return {"model_id": model_id}  # mutable-ok: Prisma where filters require a concrete object


def _build_snapshot_where(model_id: str, existing_model: _StoredModel) -> Mapping[str, object]:
    """Build a collision-resistant CAS predicate from the raw database row."""
    model_info = existing_model.model_info
    return {  # mutable-ok: Prisma CAS requires a concrete compound where object
        "model_id": model_id,
        "model_name": existing_model.model_name,
        "litellm_params": _prisma_json_equals(existing_model.litellm_params, "litellm_params"),
        "model_info": None if model_info is None else _prisma_json_equals(model_info, "model_info"),
        "blocked": existing_model.blocked,
        "updated_at": existing_model.updated_at,
        "updated_by": existing_model.updated_by,
    }


def _validate_model_snapshot(existing_model: _StoredModel, patch_data: InternalModelDeploymentPatch) -> None:
    target_model_name = patch_data.model_name if patch_data.model_name is not None else existing_model.model_name
    if "*" in existing_model.model_name or "*" in target_model_name:
        raise ProxyException(
            message="Wildcard model deployments are not supported by this endpoint.",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="model_name",
        )

    stored_litellm_params = _json_object(existing_model.litellm_params, "litellm_params")
    stored_litellm_model = stored_litellm_params.get("model")
    if isinstance(stored_litellm_model, str):
        stored_litellm_model = decrypt_value_helper(
            stored_litellm_model,
            key="model",
            exception_type="debug",
            return_original_value=True,
        )
    target_litellm_model = (
        patch_data.litellm_params.get("model", stored_litellm_model)
        if patch_data.litellm_params is not None
        else stored_litellm_model
    )
    if any(
        isinstance(model, str) and model.startswith("auto_router/")
        for model in (stored_litellm_model, target_litellm_model)
    ):
        raise ProxyException(
            message="Auto-router model deployments are not supported by this endpoint.",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="model_id",
        )

    model_info = _json_object(existing_model.model_info, "model_info")
    if model_info.get("team_id") not in (None, ""):
        raise ProxyException(
            message="Team-scoped model deployments are not supported by this endpoint.",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="model_id",
        )
    if existing_model.updated_at is None:
        raise ProxyException(
            message="Model deployment is missing updated_at and cannot be updated safely.",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param="updated_at",
        )


@router.get(
    "/internal/v1/model-deployments",
    dependencies=(Depends(user_api_key_auth),),
    include_in_schema=False,
)
async def list_internal_model_deployments(
    response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InternalModelDeploymentListResponse:
    from litellm.proxy.proxy_server import prisma_client, store_model_in_db

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={  # mutable-ok: FastAPI requires a JSON object for structured error details
                    "error": CommonProxyErrors.db_not_connected_error.value
                },
            )
        if store_model_in_db is not True:
            raise ProxyException(
                message="Model reads only supported for DB-stored models",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=None,
            )
        if not user_api_key_has_admin_view(user_api_key_dict):
            raise ProxyException(
                message="Only proxy admins and admin viewers can read model deployments through this endpoint.",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
                param=None,
            )

        model_table = _require_model_list_table(prisma_client.writer_db)
        stored_models = await model_table.find_many(order={"model_id": "asc"})
        response.headers["Cache-Control"] = "no-store"
        return InternalModelDeploymentListResponse(
            data=[
                _build_internal_model_deployment_response(stored_model.model_id, stored_model)
                for stored_model in stored_models
            ]
        )

    except Exception as exc:
        verbose_proxy_logger.exception("Error reading internal model deployments: %s", exc)
        if isinstance(exc, (HTTPException, ProxyException)):
            raise
        raise ProxyException(
            message=f"Error reading model deployments: {exc}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        ) from exc


@router.get(
    "/internal/v1/model-deployments/{model_id}",
    dependencies=(Depends(user_api_key_auth),),
    include_in_schema=False,
)
async def get_internal_model_deployment(
    model_id: str,
    response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InternalModelDeploymentResponse:
    from litellm.proxy.proxy_server import prisma_client, store_model_in_db

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={  # mutable-ok: FastAPI requires a JSON object for structured error details
                    "error": CommonProxyErrors.db_not_connected_error.value
                },
            )
        if store_model_in_db is not True:
            raise ProxyException(
                message="Model reads only supported for DB-stored models",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=None,
            )
        if not user_api_key_has_admin_view(user_api_key_dict):
            raise ProxyException(
                message="Only proxy admins and admin viewers can read model deployments through this endpoint.",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
                param=None,
            )

        model_table = _require_model_table(prisma_client.writer_db)
        stored_model = await model_table.find_unique(where=_model_id_where(model_id))
        if stored_model is None:
            raise ProxyException(
                message=f"Model {model_id} not found in database.",
                type=ProxyErrorTypes.not_found_error,
                code=status.HTTP_404_NOT_FOUND,
                param="model_id",
            )

        response.headers["Cache-Control"] = "no-store"
        return _build_internal_model_deployment_response(model_id, stored_model)

    except Exception as exc:
        verbose_proxy_logger.exception("Error reading internal model deployment: %s", exc)
        if isinstance(exc, (HTTPException, ProxyException)):
            raise
        raise ProxyException(
            message=f"Error reading model: {exc}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        ) from exc


@router.patch(
    "/internal/v1/model-deployments/{model_id}",
    dependencies=(Depends(user_api_key_auth),),
    include_in_schema=False,
)
async def patch_internal_model_deployment(
    model_id: str,
    patch_data: InternalModelDeploymentPatch,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InternalModelDeploymentPatchResponse:
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={  # mutable-ok: FastAPI requires a JSON object for structured error details
                    "error": CommonProxyErrors.db_not_connected_error.value
                },
            )
        if store_model_in_db is not True:
            raise ProxyException(
                message="Model updates only supported for DB-stored models",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=None,
            )
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise ProxyException(
                message="Only proxy admins can update model deployments through this endpoint.",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
                param=None,
            )

        model_table = _require_model_table(prisma_client.writer_db)
        existing_model: _StoredModel | None = None
        after_model: LiteLLM_ProxyModelTable | None = None
        for attempt in range(_MAX_PATCH_ATTEMPTS):
            existing_model = await model_table.find_unique(where=_model_id_where(model_id))
            if existing_model is None:
                if llm_router and llm_router.get_deployment(model_id=model_id) is not None:
                    raise ProxyException(
                        message="Cannot edit config-based model. Store model in DB via /model/new first.",
                        type=ProxyErrorTypes.validation_error.value,
                        code=status.HTTP_400_BAD_REQUEST,
                        param=None,
                    )
                raise ProxyException(
                    message=f"Model {model_id} not found on proxy.",
                    type=ProxyErrorTypes.not_found_error,
                    code=status.HTTP_404_NOT_FOUND,
                    param=None,
                )

            _validate_model_snapshot(existing_model, patch_data)

            update_data = _build_update_data(
                existing_model=existing_model,
                patch=patch_data,
                updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            )
            candidate_after_model = _build_after_model(existing_model, update_data)
            update_count = await model_table.update_many(
                where=_build_snapshot_where(model_id=model_id, existing_model=existing_model),
                data=update_data,
            )
            if update_count == 1:
                after_model = candidate_after_model
                break
            if update_count != 0:
                raise RuntimeError(f"unexpected update_many count: {update_count}")
            verbose_proxy_logger.debug(
                "Concurrent internal model deployment patch detected for %s; retrying attempt %d/%d",
                model_id,
                attempt + 1,
                _MAX_PATCH_ATTEMPTS,
            )

        if after_model is None:
            raise ProxyException(
                message="Model deployment was modified concurrently; retry the update.",
                type="conflict",
                code=status.HTTP_409_CONFLICT,
                param="model_id",
            )

        # The periodic DB sync applies the update to each process. Calling
        # clear_cache() here would remove every DB-backed deployment before
        # reloading them, creating a routing outage while this PATCH completes.

        asyncio.create_task(
            create_object_audit_log(
                object_id=model_id,
                action="updated",
                user_api_key_dict=user_api_key_dict,
                table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                before_value=_model_json(existing_model),
                after_value=_model_json(after_model),
                litellm_changed_by=user_api_key_dict.user_id,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
            )
        )
        return InternalModelDeploymentPatchResponse(model_id=model_id)

    except Exception as exc:
        verbose_proxy_logger.exception("Error in internal model deployment patch: %s", exc)
        if isinstance(exc, (HTTPException, ProxyException)):
            raise
        raise ProxyException(
            message=f"Error updating model: {exc}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        ) from exc


@router.delete(
    "/internal/v1/model-deployments/{model_id}",
    dependencies=(Depends(user_api_key_auth),),
    include_in_schema=False,
)
async def delete_internal_model_deployment(
    model_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InternalModelDeploymentDeleteResponse:
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        if store_model_in_db is not True:
            raise ProxyException(
                message="Model deletions only supported for DB-stored models",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=None,
            )
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise ProxyException(
                message="Only proxy admins can delete model deployments through this endpoint.",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
                param=None,
            )

        model_table = _require_model_delete_table(prisma_client.writer_db)
        existing_model = await model_table.find_unique(where=_model_id_where(model_id))
        if existing_model is None:
            raise ProxyException(
                message=f"Model {model_id} not found on proxy.",
                type=ProxyErrorTypes.not_found_error,
                code=status.HTTP_404_NOT_FOUND,
                param=None,
            )

        # This sidecar contract manages concrete AIGateway deployment names only.
        # Wildcard and default fallback routing are outside its supported deployment set, so their Router-derived
        # indexes are not part of this delete contract.
        if "*" in existing_model.model_name:
            raise ProxyException(
                message="Wildcard model deployments are not supported by this endpoint.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="model_name",
            )

        stored_litellm_params = _json_object(existing_model.litellm_params, "litellm_params")
        stored_litellm_model = stored_litellm_params.get("model")
        if isinstance(stored_litellm_model, str):
            stored_litellm_model = decrypt_value_helper(
                stored_litellm_model,
                key="model",
                exception_type="debug",
                return_original_value=True,
            )
        if isinstance(stored_litellm_model, str) and stored_litellm_model.startswith("auto_router/"):
            raise ProxyException(
                message="Auto-router model deployments are not supported by this endpoint.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="model_id",
            )

        model_info = _json_object(existing_model.model_info, "model_info")
        if model_info.get("team_id") not in (None, ""):
            raise ProxyException(
                message="Team-scoped model deployments are not supported by this endpoint.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="model_id",
            )

        deleted_count = await model_table.delete_many(
            where=_build_snapshot_where(model_id=model_id, existing_model=existing_model)
        )
        if deleted_count != 1:
            if deleted_count == 0:
                raise ProxyException(
                    message="Model deployment was modified concurrently; retry the delete.",
                    type="conflict",
                    code=status.HTTP_409_CONFLICT,
                    param="model_id",
                )
            raise RuntimeError(f"unexpected delete_many count: {deleted_count}")

        if llm_router is not None:
            deleted_deployment: object = llm_router.delete_deployment(id=model_id)
            deleted_name, deleted_model = _router_deployment_name_and_model(deleted_deployment)
            if (
                isinstance(deleted_name, str)
                and isinstance(deleted_model, str)
                and deleted_model.startswith("auto_router/")
            ):
                llm_router.auto_routers.pop(deleted_name, None)
                llm_router.complexity_routers.pop(deleted_name, None)
                llm_router.adaptive_routers.pop(deleted_name, None)
                llm_router.quality_routers.pop(deleted_name, None)

        asyncio.create_task(
            create_object_audit_log(
                object_id=model_id,
                action="deleted",
                user_api_key_dict=user_api_key_dict,
                table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                before_value=_model_json(existing_model),
                after_value=None,
                litellm_changed_by=user_api_key_dict.user_id,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
            )
        )
        return InternalModelDeploymentDeleteResponse(model_id=model_id)

    except Exception as exc:
        verbose_proxy_logger.exception("Error deleting internal model deployment: %s", exc)
        if isinstance(exc, (HTTPException, ProxyException)):
            raise
        raise ProxyException(
            message=f"Error deleting model: {exc}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        ) from exc
