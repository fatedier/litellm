import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError
from prisma.models import LiteLLM_ProxyModelTable as PrismaLiteLLM_ProxyModelTable

from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_ProxyModelTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.internal_model_management_endpoints import (
    InternalModelDeploymentPatch,
    _build_after_model,
    _build_internal_model_deployment_response,
    _build_snapshot_where,
    _build_update_data,
    _json_object,
    _model_json,
    _require_model_delete_table,
    _require_model_list_table,
    _require_model_table,
    delete_internal_model_deployment,
    get_internal_model_deployment,
    list_internal_model_deployments,
    patch_internal_model_deployment,
    router,
)


def _internal_model_test_client(
    user_role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN,
) -> TestClient:
    from litellm.proxy.proxy_server import openai_exception_handler

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProxyException, openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin",
        user_role=user_role,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_internal_routes_are_registered_as_hidden_management_routes() -> None:
    detail_routes = tuple(route for route in router.routes if route.path == "/internal/v1/model-deployments/{model_id}")
    collection_routes = tuple(route for route in router.routes if route.path == "/internal/v1/model-deployments")

    assert frozenset(method for route in detail_routes for method in route.methods) == frozenset(
        ("DELETE", "GET", "PATCH")
    )
    assert frozenset(method for route in collection_routes for method in route.methods) == frozenset(("GET",))
    assert all(route.include_in_schema is False for route in (*detail_routes, *collection_routes))
    assert detail_routes[0].path in LiteLLMRoutes.management_routes.value
    assert collection_routes[0].path in LiteLLMRoutes.management_routes.value
    assert all(router.routes.index(collection_routes[0]) < router.routes.index(route) for route in detail_routes)


def test_build_internal_model_deployment_response_returns_configured_state_without_secrets() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o-test",
        litellm_params={
            "model": "encrypted-model",
            "litellm_credential_name": "encrypted-credential",
            "api_key": "encrypted-secret",
            "authorization": "encrypted-authorization",
            "api_base": "os.environ/OPENAI_API_BASE",
            "future_router_option": {"enabled": True},
        },
        model_info={"base_model": "openai/gpt-4o", "future_model_option": True},
        blocked=True,
        created_at=datetime(2026, 7, 29, 8, tzinfo=timezone.utc),
        created_by="creator",
        updated_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc),
        updated_by="updater",
    )
    decrypted_values = {
        "encrypted-model": "openai/gpt-4o",
        "encrypted-credential": "openai-prod",
        "encrypted-secret": "sk-secret-value",
        "encrypted-authorization": "Bearer secret-value",
    }

    with patch(
        "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
        side_effect=lambda value, **kwargs: decrypted_values.get(value, value),
    ):
        result = _build_internal_model_deployment_response("dep-1", stored_model)

    assert result.model_dump() == {
        "model_id": "dep-1",
        "model_name": "gpt-4o-test",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "litellm_credential_name": "openai-prod",
            "authorization": "Bear***********alue",
            "api_base": "os.environ/OPENAI_API_BASE",
            "future_router_option": {"enabled": True},
        },
        "model_info": {"base_model": "openai/gpt-4o", "future_model_option": True},
        "blocked": True,
        "created_at": datetime(2026, 7, 29, 8, tzinfo=timezone.utc),
        "created_by": "creator",
        "updated_at": datetime(2026, 7, 29, 9, tzinfo=timezone.utc),
        "updated_by": "updater",
    }


def test_build_internal_model_deployment_response_preserves_null_model_info() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "openai/gpt-4o"},
        model_info=None,
    )

    with patch(
        "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
        side_effect=lambda value, **kwargs: value,
    ):
        result = _build_internal_model_deployment_response("dep-1", stored_model)

    assert result.model_info is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_role",
    (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
)
async def test_internal_get_reads_configured_state_from_writer(user_role: LitellmUserRoles) -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "tpm": 100_000},
        model_info={"base_model": "openai/gpt-4o"},
        updated_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )
    writer_table = MagicMock()
    writer_table.find_unique = AsyncMock(return_value=stored_model)
    writer_table.update_many = AsyncMock()
    reader_table = MagicMock()
    reader_table.find_unique = AsyncMock(return_value=None)
    reader_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = writer_table
    mock_prisma.db.litellm_proxymodeltable = reader_table
    mock_router = MagicMock()
    http_response = Response()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: "openai/gpt-4o" if value == "encrypted-model" else value,
        ),
    ):
        result = await get_internal_model_deployment(
            model_id="dep-1",
            response=http_response,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=user_role,
            ),
        )

    assert result.litellm_params == {"model": "openai/gpt-4o", "tpm": 100_000}
    assert result.model_info == {"base_model": "openai/gpt-4o"}
    assert http_response.headers["Cache-Control"] == "no-store"
    writer_table.find_unique.assert_awaited_once_with(where={"model_id": "dep-1"})
    reader_table.find_unique.assert_not_awaited()
    mock_router.assert_not_called()


@pytest.mark.parametrize(
    "user_role",
    (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
)
def test_internal_list_item_matches_detail_http_response_for_prisma_row(
    user_role: LitellmUserRoles,
) -> None:
    stored_model = PrismaLiteLLM_ProxyModelTable(
        model_id="dep-real",
        model_name="gpt-4o",
        litellm_params=json.dumps(
            {
                "model": "openai/gpt-4o",
                "litellm_credential_name": "openai-prod",
                "api_key": "top-level-api-secret",
                "token": "top-level-token-secret",
                "nested": {
                    "config": {"enabled": True, "weights": [1, 2, 3]},
                    "authorization": "nested-authorization-secret",
                },
            }
        ),
        model_info=json.dumps(
            {
                "base_model": "openai/gpt-4o",
                "metadata": {"level_one": {"level_two": ["value", {"enabled": True}]}},
            }
        ),
        blocked=False,
        created_at=datetime(2026, 7, 29, 8, tzinfo=timezone.utc),
        created_by="creator",
        updated_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc),
        updated_by="updater",
    )
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=[stored_model])
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: value,
        ),
    ):
        client = _internal_model_test_client(user_role)
        list_response = client.get("/internal/v1/model-deployments")
        detail_response = client.get("/internal/v1/model-deployments/dep-real")

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert list_response.headers["Cache-Control"] == "no-store"
    detail_data = detail_response.json()
    assert list_response.json() == {"data": [detail_data]}
    assert detail_data["created_at"] == "2026-07-29T08:00:00Z"
    assert detail_data["updated_at"] == "2026-07-29T09:00:00Z"
    assert detail_data["litellm_params"]["litellm_credential_name"] == "openai-prod"
    assert "api_key" not in detail_data["litellm_params"]
    assert detail_data["litellm_params"]["token"] == "top-**************cret"
    assert detail_data["litellm_params"]["nested"]["authorization"] == "nest*******************cret"
    assert detail_data["litellm_params"]["nested"]["config"] == {
        "enabled": True,
        "weights": [1, 2, 3],
    }
    model_table.find_many.assert_awaited_once_with(order={"model_id": "asc"})
    model_table.find_unique.assert_awaited_once_with(where={"model_id": "dep-real"})


@pytest.mark.asyncio
async def test_internal_list_returns_empty_data_without_pagination() -> None:
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=[])
    model_table.find_unique = AsyncMock()
    model_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        result = await list_internal_model_deployments(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert result.model_dump() == {"data": []}
    model_table.find_many.assert_awaited_once_with(order={"model_id": "asc"})


def test_internal_list_and_detail_preserve_json_null_semantics() -> None:
    stored_model = SimpleNamespace(
        model_id="dep-null",
        model_name="null-json",
        litellm_params=None,
        model_info=None,
        blocked=False,
        created_at=None,
        created_by=None,
        updated_at=None,
        updated_by=None,
    )
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=[stored_model])
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        client = _internal_model_test_client()
        list_response = client.get("/internal/v1/model-deployments")
        detail_response = client.get("/internal/v1/model-deployments/dep-null")

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert list_response.json() == {"data": [detail_response.json()]}
    assert detail_response.json()["litellm_params"] == {}
    assert detail_response.json()["model_info"] is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("litellm_params", "{not-valid-json"),
        ("model_info", ["not", "an", "object"]),
    ),
)
def test_internal_list_and_detail_reject_invalid_stored_json(
    field_name: str,
    invalid_value: object,
) -> None:
    stored_model = SimpleNamespace(
        model_id="dep-invalid",
        model_name="invalid-json",
        litellm_params=invalid_value if field_name == "litellm_params" else {"model": "openai/gpt-4o"},
        model_info=invalid_value if field_name == "model_info" else {"base_model": "openai/gpt-4o"},
        blocked=False,
        created_at=None,
        created_by=None,
        updated_at=None,
        updated_by=None,
    )
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=[stored_model])
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        client = _internal_model_test_client()
        list_response = client.get("/internal/v1/model-deployments")
        detail_response = client.get("/internal/v1/model-deployments/dep-invalid")

    assert list_response.status_code == 500
    assert detail_response.status_code == 500
    assert f"stored {field_name} must be a JSON object" in list_response.json()["error"]["message"]
    assert f"stored {field_name} must be a JSON object" in detail_response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_internal_list_returns_db_not_connected_error() -> None:
    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),
        pytest.raises(HTTPException) as exc_info,
    ):
        await list_internal_model_deployments(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {"error": CommonProxyErrors.db_not_connected_error.value}


@pytest.mark.asyncio
async def test_internal_list_rejects_non_admin_before_query() -> None:
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        pytest.raises(ProxyException) as exc_info,
    ):
        await list_internal_model_deployments(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user",
                user_role=LitellmUserRoles.INTERNAL_USER,
            ),
        )

    assert exc_info.value.code == "403"
    model_table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_list_rejects_when_db_storage_is_disabled() -> None:
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", False),
        pytest.raises(ProxyException) as exc_info,
    ):
        await list_internal_model_deployments(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert exc_info.value.code == "400"
    model_table.find_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_role", "stored_model", "expected_code"),
    (
        (LitellmUserRoles.INTERNAL_USER, MagicMock(), 403),
        (LitellmUserRoles.PROXY_ADMIN, None, 404),
    ),
)
async def test_internal_get_rejects_non_admin_and_missing_model(
    user_role: LitellmUserRoles,
    stored_model: object | None,
    expected_code: int,
) -> None:
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.update_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        pytest.raises(ProxyException) as exc_info,
    ):
        await get_internal_model_deployment(
            model_id="dep-1",
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user",
                user_role=user_role,
            ),
        )

    assert exc_info.value.code == str(expected_code)
    if user_role == LitellmUserRoles.INTERNAL_USER:
        model_table.find_unique.assert_not_awaited()
    else:
        model_table.find_unique.assert_awaited_once()


@pytest.mark.asyncio
async def test_internal_delete_uses_writer_and_cleans_router_and_audit() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "openai/gpt-4o", "api_key": "encrypted-secret"},
        model_info={"id": "dep-1"},
        updated_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )
    writer_table = MagicMock()
    writer_table.find_unique = AsyncMock(return_value=stored_model)
    writer_table.delete_many = AsyncMock(return_value=1)
    reader_table = MagicMock()
    reader_table.find_unique = AsyncMock(return_value=None)
    reader_table.delete_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = writer_table
    mock_prisma.db.litellm_proxymodeltable = reader_table
    mock_router = MagicMock()
    mock_router.delete_deployment.return_value = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o"},
    }
    audit = AsyncMock(return_value=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
    ):
        result = await delete_internal_model_deployment(
            model_id="dep-1",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )
        await asyncio.sleep(0)

    assert result.model_id == "dep-1"
    writer_table.find_unique.assert_awaited_once_with(where={"model_id": "dep-1"})
    writer_table.delete_many.assert_awaited_once_with(where=_build_snapshot_where("dep-1", stored_model))
    reader_table.find_unique.assert_not_awaited()
    reader_table.delete_many.assert_not_awaited()
    mock_router.delete_deployment.assert_called_once_with(id="dep-1")
    audit.assert_awaited_once()
    assert json.loads(audit.await_args.kwargs["before_value"])["model_id"] == "dep-1"
    assert audit.await_args.kwargs["after_value"] is None


@pytest.mark.asyncio
async def test_internal_delete_returns_conflict_when_snapshot_changes() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "openai/gpt-4o"},
        model_info={"id": "dep-1"},
        blocked=False,
        updated_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        updated_by="before-admin",
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.delete_many = AsyncMock(return_value=0)
    mock_router = MagicMock()
    audit = AsyncMock(return_value=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
        pytest.raises(ProxyException) as exc_info,
    ):
        await delete_internal_model_deployment(
            model_id="dep-1",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.code == "409"
    assert exc_info.value.param == "model_id"
    model_table.delete_many.assert_awaited_once_with(where=_build_snapshot_where("dep-1", stored_model))
    mock_router.delete_deployment.assert_not_called()
    audit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_role", "stored_model", "store_model_in_db", "expected_code"),
    (
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, MagicMock(), True, 403),
        (LitellmUserRoles.PROXY_ADMIN, None, True, 404),
        (LitellmUserRoles.PROXY_ADMIN, MagicMock(), False, 400),
    ),
)
async def test_internal_delete_rejects_unsupported_requests(
    user_role: LitellmUserRoles,
    stored_model: object | None,
    store_model_in_db: bool,
    expected_code: int,
) -> None:
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.delete_many = AsyncMock(return_value=1)
    mock_router = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", store_model_in_db),
        pytest.raises(ProxyException) as exc_info,
    ):
        await delete_internal_model_deployment(
            model_id="dep-1",
            user_api_key_dict=UserAPIKeyAuth(user_id="user", user_role=user_role),
        )

    assert exc_info.value.code == str(expected_code)
    if user_role == LitellmUserRoles.PROXY_ADMIN and store_model_in_db:
        model_table.find_unique.assert_awaited_once()
    else:
        model_table.find_unique.assert_not_awaited()
    model_table.delete_many.assert_not_awaited()
    mock_router.delete_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_internal_delete_rejects_wildcard_model_before_writing() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-wildcard",
        model_name="public/*",
        litellm_params={"model": "openai/*"},
        model_info={"id": "dep-wildcard"},
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.delete_many = AsyncMock()
    mock_router = MagicMock()
    audit = AsyncMock(return_value=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
        pytest.raises(ProxyException) as exc_info,
    ):
        await delete_internal_model_deployment(
            model_id="dep-wildcard",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "model_name"
    model_table.delete_many.assert_not_awaited()
    mock_router.delete_deployment.assert_not_called()
    audit.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_delete_rejects_auto_router_model_before_writing() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-auto-router",
        model_name="smart-router",
        litellm_params={"model": "encrypted-auto-router"},
        model_info={"id": "dep-auto-router"},
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.delete_many = AsyncMock()
    mock_router = MagicMock()
    audit = AsyncMock(return_value=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            return_value="auto_router/custom",
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
        pytest.raises(ProxyException) as exc_info,
    ):
        await delete_internal_model_deployment(
            model_id="dep-auto-router",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "model_id"
    model_table.delete_many.assert_not_awaited()
    mock_router.delete_deployment.assert_not_called()
    audit.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_delete_rejects_team_scoped_model() -> None:
    stored_model = LiteLLM_ProxyModelTable(
        model_id="dep-team",
        model_name="gpt-4o",
        litellm_params={"model": "openai/gpt-4o"},
        model_info={"id": "dep-team", "team_id": "team-1"},
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=stored_model)
    model_table.delete_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        pytest.raises(ProxyException) as exc_info,
    ):
        await delete_internal_model_deployment(
            model_id="dep-team",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.code == "400"
    model_table.delete_many.assert_not_awaited()


def test_build_update_data_only_changes_specified_litellm_params() -> None:
    existing = SimpleNamespace(
        litellm_params={
            "model": "encrypted-model",
            "tpm": 100_000,
            "use_in_pass_through": True,
            "use_litellm_proxy": True,
        },
        model_info={"id": "dep-1", "db_model": True},
    )
    patch_data = InternalModelDeploymentPatch(litellm_params={"tpm": 200_000})

    with patch(
        "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
        side_effect=lambda value: value,
    ):
        update_data = _build_update_data(existing, patch_data, "admin")

    params = json.loads(update_data["litellm_params"])
    assert params == {
        "model": "encrypted-model",
        "tpm": 200_000,
        "use_in_pass_through": True,
        "use_litellm_proxy": True,
    }
    assert "model_info" not in update_data
    assert "model_name" not in update_data
    assert "blocked" not in update_data


def test_build_update_data_preserves_arbitrary_litellm_params() -> None:
    existing = SimpleNamespace(
        litellm_params={"model": "encrypted-model", "weight": 100},
        model_info={"id": "dep-1"},
    )
    patch_data = InternalModelDeploymentPatch(
        litellm_params={
            "weight": 600,
            "future_router_option": {"enabled": True},
        }
    )

    with patch(
        "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
        side_effect=lambda value: value,
    ):
        update_data = _build_update_data(existing, patch_data, "admin")

    params = json.loads(update_data["litellm_params"])
    assert params["weight"] == 600
    assert params["future_router_option"] == {"enabled": True}


def test_build_update_data_applies_direct_tombstones_and_replaces_nested_objects() -> None:
    existing = SimpleNamespace(
        litellm_params={
            "model": "encrypted-model",
            "api_key": "encrypted-api-key",
            "api_base": "encrypted-api-base",
            "complexity_router_config": {"threshold": 0.5, "keep": True},
        },
        model_info={"id": "dep-1"},
    )
    patch_data = InternalModelDeploymentPatch(
        litellm_params={
            "api_key": None,
            "api_base": "https://new.example.com",
            "complexity_router_config": {"threshold": None},
        }
    )
    decrypted_values = {
        "encrypted-model": "openai/gpt-4o",
        "encrypted-api-key": "old-secret",
        "encrypted-api-base": "https://old.example.com",
    }

    with (
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: decrypted_values.get(value, value),
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
            return_value="encrypted-new-api-base",
        ) as encrypt,
    ):
        update_data = _build_update_data(existing, patch_data, "admin")

    assert json.loads(update_data["litellm_params"]) == {
        "model": "encrypted-model",
        "api_base": "encrypted-new-api-base",
        "complexity_router_config": {"threshold": None},
    }
    encrypt.assert_called_once_with("https://new.example.com")


def test_build_update_data_removes_special_pricing_from_both_blobs() -> None:
    existing = SimpleNamespace(
        litellm_params={
            "model": "encrypted-model",
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
        },
        model_info={
            "id": "dep-1",
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
        },
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(litellm_params={"input_cost_per_token": None}),
        "admin",
    )

    assert json.loads(update_data["litellm_params"]) == {
        "model": "encrypted-model",
        "output_cost_per_token": 0.2,
    }
    assert json.loads(update_data["model_info"]) == {
        "id": "dep-1",
        "output_cost_per_token": 0.2,
    }


@pytest.mark.parametrize(
    ("historical_field", "historical_value"),
    [
        ("forward_client_headers", {}),
        ("nova_cost_discount", 1.5),
    ],
)
def test_build_update_data_preserves_invalid_historical_litellm_params_when_deleting_credential(
    historical_field,
    historical_value,
) -> None:
    existing = SimpleNamespace(
        litellm_params={
            "model": "openai/gpt-4o",
            "litellm_credential_name": "credential-before",
            historical_field: historical_value,
        },
        model_info={"id": "dep-1"},
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(litellm_params={"litellm_credential_name": None}),
        "admin",
    )

    params = json.loads(update_data["litellm_params"])
    assert "litellm_credential_name" not in params
    assert params[historical_field] == historical_value


def test_build_update_data_preserves_model_info_identity() -> None:
    existing = SimpleNamespace(
        litellm_params={"model": "encrypted-model"},
        model_info={
            "id": "dep-1",
            "db_model": True,
            "mode": "chat",
            "input_cost_per_token": 0.1,
        },
    )
    patch_data = InternalModelDeploymentPatch(model_info={"base_model": "azure/gpt-4o"})

    update_data = _build_update_data(existing, patch_data, "admin")

    assert json.loads(update_data["model_info"]) == {
        "id": "dep-1",
        "db_model": True,
        "mode": "chat",
        "input_cost_per_token": 0.1,
        "base_model": "azure/gpt-4o",
    }
    assert "litellm_params" not in update_data


@pytest.mark.parametrize(
    ("existing_model_info", "expected_model_info"),
    [
        ({"id": "dep-1", "base_model": "openai/gpt-4o", "future_option": True}, {"id": "dep-1", "future_option": True}),
        (None, {}),
    ],
)
def test_build_update_data_removes_base_model_without_dumping_model_info_defaults(
    existing_model_info,
    expected_model_info,
) -> None:
    existing = SimpleNamespace(
        litellm_params={"model": "encrypted-model"},
        model_info=existing_model_info,
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(model_info={"base_model": None}),
        "admin",
    )

    assert json.loads(update_data["model_info"]) == expected_model_info
    assert "litellm_params" not in update_data


def test_build_update_data_rejects_missing_stored_model_before_encryption() -> None:
    existing = SimpleNamespace(
        litellm_params={"tpm": 100_000},
        model_info={"id": "dep-1"},
    )

    with (
        patch("litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper") as encrypt,
        pytest.raises(TypeError, match="stored litellm_params.model must be a non-empty string"),
    ):
        _build_update_data(
            existing,
            InternalModelDeploymentPatch(litellm_params={"api_base": "https://new.example.com"}),
            "admin",
        )

    encrypt.assert_not_called()


def test_build_update_data_preserves_invalid_historical_model_info_when_deleting_base_model() -> None:
    existing = SimpleNamespace(
        litellm_params={"model": "encrypted-model"},
        model_info={
            "id": "dep-1",
            "base_model": "openai/gpt-4o",
            "nova_cost_discount": 1.5,
        },
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(model_info={"base_model": None}),
        "admin",
    )

    assert json.loads(update_data["model_info"]) == {
        "id": "dep-1",
        "nova_cost_discount": 1.5,
    }
    assert "litellm_params" not in update_data


def test_build_update_data_ignores_invalid_historical_litellm_params_when_deleting_base_model() -> None:
    existing = SimpleNamespace(
        litellm_params={
            "model": "encrypted-model",
            "nova_cost_discount": 1.5,
        },
        model_info={
            "id": "dep-1",
            "base_model": "openai/gpt-4o",
        },
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(model_info={"base_model": None}),
        "admin",
    )

    assert json.loads(update_data["model_info"]) == {"id": "dep-1"}
    assert "litellm_params" not in update_data


def test_build_update_data_advances_timestamp_past_existing_value() -> None:
    existing_updated_at = datetime(2026, 7, 27, 12, 0, 0, 123_000, tzinfo=timezone.utc)
    existing = SimpleNamespace(
        updated_at=existing_updated_at,
        litellm_params={"model": "encrypted-model"},
        model_info={"id": "dep-1"},
    )

    with patch(
        "litellm.proxy.management_endpoints.internal_model_management_endpoints.get_utc_datetime",
        return_value=existing_updated_at,
    ):
        update_data = _build_update_data(
            existing,
            InternalModelDeploymentPatch(blocked=True),
            "admin",
        )

    assert update_data["updated_at"] == existing_updated_at + timedelta(milliseconds=1)


def test_build_update_data_preserves_explicit_false() -> None:
    existing = SimpleNamespace(
        litellm_params={"model": "encrypted-model"},
        model_info={"id": "dep-1"},
    )

    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(blocked=False),
        "admin",
    )

    assert update_data["blocked"] is False


def test_build_after_model_preserves_null_model_info_when_unmodified() -> None:
    existing = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model"},
        model_info=None,
        updated_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    update_data = _build_update_data(
        existing,
        InternalModelDeploymentPatch(blocked=True),
        "admin",
    )

    after_model = _build_after_model(existing, update_data)
    after_json = _model_json(after_model)

    assert "model_info" not in update_data
    assert after_model.model_info is None
    assert after_json is not None
    after_value = json.loads(after_json)
    assert "model_info" not in after_value


def test_model_json_rejects_non_string_serialization() -> None:
    model = SimpleNamespace(model_dump_json=MagicMock(return_value={"model_id": "dep-1"}))

    assert _model_json(model) is None


def test_json_object_rejects_non_object_json() -> None:
    with pytest.raises(TypeError, match="stored model_info must be a JSON object"):
        _json_object('["not-an-object"]', "model_info")


def test_require_model_table_rejects_incomplete_table() -> None:
    incomplete_table = SimpleNamespace(find_unique=AsyncMock())
    writer_db = SimpleNamespace(litellm_proxymodeltable=incomplete_table)

    with pytest.raises(TypeError, match="writer database does not expose the model deployment table"):
        _require_model_table(writer_db)


def test_require_model_delete_table_requires_delete_many() -> None:
    detail_table = SimpleNamespace(
        find_unique=AsyncMock(),
        update_many=AsyncMock(),
    )
    writer_db = SimpleNamespace(litellm_proxymodeltable=detail_table)

    with pytest.raises(TypeError, match="writer database does not expose the model deployment delete table"):
        _require_model_delete_table(writer_db)


def test_require_model_list_table_adds_find_many_requirement() -> None:
    detail_table = SimpleNamespace(
        find_unique=AsyncMock(),
        update_many=AsyncMock(),
    )
    writer_db = SimpleNamespace(litellm_proxymodeltable=detail_table)

    assert _require_model_table(writer_db) is detail_table
    with pytest.raises(TypeError, match="writer database does not expose the model deployment list table"):
        _require_model_list_table(writer_db)


def test_build_snapshot_where_matches_raw_database_snapshot() -> None:
    existing_updated_at = datetime(2026, 7, 27, 12, tzinfo=timezone.utc)
    existing = SimpleNamespace(
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "weight": 100},
        model_info={"id": "dep-1", "base_model": "gpt-4o"},
        blocked=False,
        updated_at=existing_updated_at,
        updated_by="admin",
    )

    where = _build_snapshot_where("dep-1", existing)

    assert where["model_id"] == "dep-1"
    assert where["model_name"] == "gpt-4o"
    assert where["litellm_params"]["equals"].data == {
        "model": "encrypted-model",
        "weight": 100,
    }
    assert where["model_info"]["equals"].data == {"id": "dep-1", "base_model": "gpt-4o"}
    assert where["blocked"] is False
    assert where["updated_at"] == existing_updated_at
    assert where["updated_by"] == "admin"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_name": None},
        {"blocked": None},
        {"litellm_params": {}},
        {"litellm_params": None},
        {"model_info": {}},
        {"model_info": None},
        {"model_info": {"base_model": ""}},
        {"model_info": {"base_model": "gpt-4o", "id": "replacement"}},
        {"litellm_params": {"nova_cost_discount": 1.5}},
        {"litellm_params": {"model": {}}},
        {"model_info": {"nova_cost_discount": 0.8}},
        {"model_info": {"input_cost_per_token": 0.000001}},
        {"model_info": {"base_model": {}}},
        {"model_info": {"future_model_option": {"enabled": True}}},
        {"unknown": True},
    ],
)
def test_patch_validation_rejects_empty_null_and_unknown_fields(payload) -> None:
    with pytest.raises(ValidationError):
        InternalModelDeploymentPatch.model_validate(payload)


def test_model_tombstone_returns_stable_422_error() -> None:
    response = _internal_model_test_client().patch(
        "/internal/v1/model-deployments/dep-1",
        json={"litellm_params": {"model": None}},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, removing litellm_params fields is not supported: model"
    )


@pytest.mark.parametrize(
    "litellm_params",
    [
        {"forward_client_headers": {}},
        {"use_xai_oauth": {}},
        {"itpm": {}},
        {"model": ""},
        {"model": {}},
    ],
)
def test_invalid_litellm_params_sets_return_422(litellm_params) -> None:
    response = _internal_model_test_client().patch(
        "/internal/v1/model-deployments/dep-1",
        json={"litellm_params": litellm_params},
    )

    assert response.status_code == 422


def test_patch_validation_ignores_unknown_top_level_fields() -> None:
    patch_data = InternalModelDeploymentPatch.model_validate(
        {
            "blocked": True,
            "future_top_level_field": "ignored",
        }
    )

    assert patch_data.model_dump(exclude_unset=True) == {"blocked": True}


def test_patch_validation_preserves_unknown_litellm_params_without_model_info_defaults() -> None:
    patch_data = InternalModelDeploymentPatch.model_validate(
        {
            "litellm_params": {
                "nova_cost_discount": 0.8,
                "future_router_option": {"enabled": True},
            },
            "model_info": {"base_model": "azure/gpt-4o"},
        }
    )

    assert patch_data.model_dump(exclude_unset=True) == {
        "litellm_params": {
            "nova_cost_discount": 0.8,
            "future_router_option": {"enabled": True},
        },
        "model_info": {"base_model": "azure/gpt-4o"},
    }


@pytest.mark.parametrize(
    ("model_id", "litellm_params", "model_info", "payload", "updated_json_field", "expected_json"),
    [
        (
            "dep-forward-headers",
            {
                "model": "openai/gpt-4o",
                "forward_client_headers": {},
                "litellm_credential_name": "credential-before",
            },
            {"id": "dep-forward-headers"},
            {"litellm_params": {"litellm_credential_name": None}},
            "litellm_params",
            {
                "model": "openai/gpt-4o",
                "forward_client_headers": {},
            },
        ),
        (
            "dep-model-info-discount",
            {"model": "openai/gpt-4o"},
            {
                "id": "dep-model-info-discount",
                "base_model": "openai/gpt-4o",
                "nova_cost_discount": 1.5,
            },
            {"model_info": {"base_model": None}},
            "model_info",
            {
                "id": "dep-model-info-discount",
                "nova_cost_discount": 1.5,
            },
        ),
    ],
)
def test_internal_patch_route_preserves_invalid_historical_fields(
    model_id,
    litellm_params,
    model_info,
    payload,
    updated_json_field,
    expected_json,
) -> None:
    existing_model = LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name="gpt-4o",
        litellm_params=litellm_params,
        model_info=model_info,
        blocked=False,
        updated_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        updated_by="before-admin",
    )
    model_table = MagicMock()
    model_table.find_unique = AsyncMock(return_value=existing_model)
    model_table.update_many = AsyncMock(return_value=1)
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: value,
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=AsyncMock(return_value=None),
        ),
    ):
        response = _internal_model_test_client().patch(
            f"/internal/v1/model-deployments/{model_id}",
            json=payload,
        )

    assert response.status_code == 200
    assert response.json() == {"model_id": model_id}
    model_table.update_many.assert_awaited_once()
    update_data = model_table.update_many.await_args.kwargs["data"]
    assert json.loads(update_data[updated_json_field]) == expected_json


def test_internal_patch_route_requires_valid_stored_model_or_model_repair() -> None:
    existing_model = LiteLLM_ProxyModelTable(
        model_id="dep-missing-model",
        model_name="gpt-4o",
        litellm_params={"tpm": 100_000},
        model_info={"id": "dep-missing-model"},
        blocked=False,
        updated_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        updated_by="before-admin",
    )
    model_table = MagicMock()
    model_table.find_unique = AsyncMock(return_value=existing_model)
    model_table.update_many = AsyncMock(return_value=1)
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = model_table
    audit = AsyncMock(return_value=None)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: value,
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
            side_effect=lambda value: f"encrypted:{value}",
        ) as encrypt,
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
    ):
        client = _internal_model_test_client()
        missing_model_response = client.patch(
            "/internal/v1/model-deployments/dep-missing-model",
            json={"blocked": False},
        )

        assert missing_model_response.status_code == 500
        assert missing_model_response.json() == {
            "error": {
                "message": "Error updating model: stored litellm_params.model must be a non-empty string",
                "type": "internal_server_error",
                "param": None,
                "code": "500",
            }
        }
        model_table.update_many.assert_not_awaited()

        repair_response = client.patch(
            "/internal/v1/model-deployments/dep-missing-model",
            json={"litellm_params": {"model": "openai/gpt-4o"}},
        )

    assert repair_response.status_code == 200
    assert repair_response.json() == {"model_id": "dep-missing-model"}
    model_table.update_many.assert_awaited_once()
    assert json.loads(model_table.update_many.await_args.kwargs["data"]["litellm_params"]) == {
        "tpm": 100_000,
        "model": "encrypted:openai/gpt-4o",
    }
    encrypt.assert_called_once_with("openai/gpt-4o")


@pytest.mark.asyncio
async def test_internal_patch_uses_writer_for_cas_without_reloading_router() -> None:
    existing_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={
            "model": "encrypted-model",
            "api_key": "encrypted-secret",
            "tpm": 100_000,
            "use_in_pass_through": True,
        },
        model_info={"id": "dep-1", "db_model": True},
        updated_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
        updated_by="before-admin",
    )

    writer_table = MagicMock()
    writer_table.find_unique = AsyncMock(return_value=existing_model)
    writer_table.update_many = AsyncMock(return_value=1)
    reader_table = MagicMock()
    reader_table.find_unique = AsyncMock(return_value=None)
    reader_table.update_many = AsyncMock()
    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable = writer_table
    mock_prisma.db.litellm_proxymodeltable = reader_table
    mock_router = MagicMock()
    audit = AsyncMock(return_value=None)
    decrypted_values = {
        "encrypted-model": "openai/gpt-4o",
        "encrypted-secret": "plain-secret",
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: decrypted_values.get(value, value),
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
            side_effect=lambda value: value,
        ) as encrypt,
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=audit,
        ),
    ):
        result = await patch_internal_model_deployment(
            model_id="dep-1",
            patch_data=InternalModelDeploymentPatch(
                litellm_params={"api_key": None, "tpm": 200_000},
                model_info={"base_model": "azure/gpt-4o"},
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )
        await asyncio.sleep(0)

    assert result.model_id == "dep-1"
    update = writer_table.update_many.await_args.kwargs
    assert update["where"] == {
        **_build_snapshot_where("dep-1", existing_model),
    }
    assert json.loads(update["data"]["litellm_params"]) == {
        "model": "encrypted-model",
        "tpm": 200_000,
        "use_in_pass_through": True,
    }
    assert json.loads(update["data"]["model_info"]) == {
        "id": "dep-1",
        "db_model": True,
        "base_model": "azure/gpt-4o",
    }
    assert mock_router.mock_calls == []
    audit.assert_awaited_once()
    audit_before_raw = audit.await_args.kwargs["before_value"]
    audit_after_raw = audit.await_args.kwargs["after_value"]
    assert audit_before_raw is not None
    assert audit_after_raw is not None
    audit_before = json.loads(audit_before_raw)
    audit_after = json.loads(audit.await_args.kwargs["after_value"])
    assert audit_before["litellm_params"]["api_key"] == "encrypted-secret"
    assert "api_key" not in audit_after["litellm_params"]
    assert "plain-secret" not in audit_before_raw
    assert "plain-secret" not in audit_after_raw
    assert audit_after["litellm_params"]["tpm"] == 200_000
    assert audit_after["model_info"]["base_model"] == "azure/gpt-4o"
    assert audit_after["updated_by"] == "admin"
    writer_table.find_unique.assert_awaited_once()
    reader_table.find_unique.assert_not_awaited()
    reader_table.update_many.assert_not_awaited()
    encrypt.assert_not_called()


@pytest.mark.asyncio
async def test_internal_patch_retries_against_newer_snapshot_after_conflict() -> None:
    first_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "tpm": 100_000, "rpm": 10},
        model_info={"id": "dep-1"},
        updated_at=datetime(2026, 7, 27, 12, tzinfo=timezone.utc),
    )
    newer_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "tpm": 200_000, "rpm": 30},
        model_info={"id": "dep-1"},
        updated_at=datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc),
    )

    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable.find_unique = AsyncMock(side_effect=[first_model, newer_model])
    mock_prisma.writer_db.litellm_proxymodeltable.update_many = AsyncMock(side_effect=[0, 1])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
            side_effect=lambda value: value,
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await patch_internal_model_deployment(
            model_id="dep-1",
            patch_data=InternalModelDeploymentPatch(litellm_params={"rpm": None}),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert result.model_id == "dep-1"
    calls = mock_prisma.writer_db.litellm_proxymodeltable.update_many.await_args_list
    assert calls[0].kwargs["where"]["updated_at"] == first_model.updated_at
    assert calls[1].kwargs["where"]["updated_at"] == newer_model.updated_at
    assert json.loads(calls[0].kwargs["data"]["litellm_params"]) == {
        "model": "encrypted-model",
        "tpm": 100_000,
    }
    assert json.loads(calls[1].kwargs["data"]["litellm_params"]) == {
        "model": "encrypted-model",
        "tpm": 200_000,
    }
    assert mock_prisma.writer_db.litellm_proxymodeltable.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_internal_patch_retries_when_json_changes_without_timestamp_change() -> None:
    timestamp = datetime(2026, 7, 27, 12, tzinfo=timezone.utc)
    first_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "tpm": 100_000, "rpm": 10},
        model_info={"id": "dep-1"},
        updated_at=timestamp,
        updated_by="before-admin",
    )
    newer_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="gpt-4o",
        litellm_params={"model": "encrypted-model", "tpm": 200_000, "rpm": 10},
        model_info={"id": "dep-1"},
        updated_at=timestamp,
        updated_by="other-admin",
    )

    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable.find_unique = AsyncMock(side_effect=[first_model, newer_model])
    mock_prisma.writer_db.litellm_proxymodeltable.update_many = AsyncMock(side_effect=[0, 1])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.encrypt_value_helper",
            side_effect=lambda value: value,
        ),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.create_object_audit_log",
            new=AsyncMock(return_value=None),
        ),
    ):
        result = await patch_internal_model_deployment(
            model_id="dep-1",
            patch_data=InternalModelDeploymentPatch(litellm_params={"rpm": 20}),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin",
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

    assert result.model_id == "dep-1"
    calls = mock_prisma.writer_db.litellm_proxymodeltable.update_many.await_args_list
    assert calls[0].kwargs["where"]["updated_at"] == timestamp
    assert calls[0].kwargs["where"]["litellm_params"]["equals"].data["tpm"] == 100_000
    assert calls[1].kwargs["where"]["updated_at"] == timestamp
    assert calls[1].kwargs["where"]["litellm_params"]["equals"].data["tpm"] == 200_000
    assert json.loads(calls[1].kwargs["data"]["litellm_params"]) == {
        "model": "encrypted-model",
        "tpm": 200_000,
        "rpm": 20,
    }


@pytest.mark.asyncio
async def test_internal_patch_returns_conflict_after_retry_exhaustion() -> None:
    existing_models = [
        LiteLLM_ProxyModelTable(
            model_id="dep-1",
            model_name="gpt-4o",
            litellm_params={"model": "encrypted-model", "tpm": 100_000},
            model_info={"id": "dep-1"},
            updated_at=datetime(2026, 7, 27, 12, minute, tzinfo=timezone.utc),
        )
        for minute in range(3)
    ]

    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable.find_unique = AsyncMock(side_effect=existing_models)
    mock_prisma.writer_db.litellm_proxymodeltable.update_many = AsyncMock(side_effect=[0, 0, 0])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "proxy-admin"),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await patch_internal_model_deployment(
                model_id="dep-1",
                patch_data=InternalModelDeploymentPatch(litellm_params={"rpm": 20}),
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="admin",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                ),
            )

    assert exc_info.value.code == "409"
    assert mock_prisma.writer_db.litellm_proxymodeltable.update_many.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_role", "team_id", "expected_code"),
    [
        (LitellmUserRoles.INTERNAL_USER, None, 403),
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, None, 403),
        (LitellmUserRoles.PROXY_ADMIN, "team-1", 400),
    ],
)
async def test_internal_patch_rejects_non_admin_and_team_models(user_role, team_id, expected_code) -> None:
    existing_model = MagicMock()
    existing_model.litellm_params = {"model": "encrypted-model"}
    existing_model.model_info = {"id": "dep-1", "team_id": team_id}

    mock_prisma = MagicMock()
    mock_prisma.writer_db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=existing_model)
    mock_prisma.writer_db.litellm_proxymodeltable.update_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await patch_internal_model_deployment(
                model_id="dep-1",
                patch_data=InternalModelDeploymentPatch(blocked=True),
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="user",
                    user_role=user_role,
                ),
            )

    assert exc_info.value.code == str(expected_code)
    if user_role != LitellmUserRoles.PROXY_ADMIN:
        mock_prisma.writer_db.litellm_proxymodeltable.find_unique.assert_not_awaited()
    mock_prisma.writer_db.litellm_proxymodeltable.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_litellm_model", "patch_data"),
    [
        ("auto_router/custom", InternalModelDeploymentPatch(blocked=True)),
        (
            "openai/gpt-4o",
            InternalModelDeploymentPatch(litellm_params={"model": "auto_router/custom"}),
        ),
    ],
)
async def test_internal_patch_rejects_auto_router_deployments_before_writing(
    stored_litellm_model,
    patch_data,
) -> None:
    existing_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name="smart-router",
        litellm_params={"model": stored_litellm_model},
        model_info={"id": "dep-1"},
        updated_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=existing_model)
    model_table.update_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.internal_model_management_endpoints.decrypt_value_helper",
            side_effect=lambda value, **kwargs: value,
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await patch_internal_model_deployment(
                model_id="dep-1",
                patch_data=patch_data,
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="admin",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                ),
            )

    assert exc_info.value.code == "400"
    model_table.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_model_name", "patch_data"),
    [
        ("public/*", InternalModelDeploymentPatch(blocked=True)),
        (
            "public/*",
            InternalModelDeploymentPatch(litellm_params={"litellm_credential_name": "new-credential"}),
        ),
        ("public/*", InternalModelDeploymentPatch(model_name="production/*")),
        ("public/*", InternalModelDeploymentPatch(model_name="production")),
        ("public", InternalModelDeploymentPatch(model_name="production/*")),
    ],
)
async def test_internal_patch_rejects_wildcard_updates_before_writing(
    stored_model_name,
    patch_data,
) -> None:
    existing_model = LiteLLM_ProxyModelTable(
        model_id="dep-1",
        model_name=stored_model_name,
        litellm_params={"model": "openai/gpt-4o"},
        model_info={"id": "dep-1"},
        updated_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    mock_prisma = MagicMock()
    model_table = mock_prisma.writer_db.litellm_proxymodeltable
    model_table.find_unique = AsyncMock(return_value=existing_model)
    model_table.update_many = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await patch_internal_model_deployment(
                model_id="dep-1",
                patch_data=patch_data,
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="admin",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                ),
            )

    assert exc_info.value.code == "400"
    model_table.update_many.assert_not_awaited()
