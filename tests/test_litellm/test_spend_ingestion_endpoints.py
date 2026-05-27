import json
import sys
import types
from types import SimpleNamespace

import litellm
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.spend_tracking import spend_ingestion_endpoints


class _FakeSpendLogsTable:
    def __init__(self, existing_spend_log=None):
        self.existing_spend_log = existing_spend_log

    async def find_unique(self, where):
        return self.existing_spend_log


class _FakePrismaClient:
    def __init__(self):
        self.db = SimpleNamespace(litellm_spendlogs=_FakeSpendLogsTable())


def _get_test_client(monkeypatch, master_key="sk-master"):
    monkeypatch.setattr(
        spend_ingestion_endpoints,
        "_get_configured_master_key",
        lambda: master_key,
    )
    app = FastAPI()
    app.include_router(spend_ingestion_endpoints.router)
    return TestClient(app)


def _payload(call_type="_arealtime"):
    return {
        "request_id": "req-123",
        "api_key_hash": "hashed-key",
        "call_type": call_type,
        "model": "gpt-realtime-1.5",
        "spend": 0.123,
        "start_time": "2026-05-18T01:00:00Z",
        "end_time": "2026-05-18T01:00:03Z",
        "session_id": "sess-123",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "input_seconds": 1.5,
            "output_seconds": 2.5,
        },
    }


def _install_fake_proxy_server(monkeypatch, proxy_server):
    if not hasattr(proxy_server, "general_settings"):
        proxy_server.general_settings = {}
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server)


def _install_fake_auth_checks(monkeypatch, get_key_object, cache_key_object=None):
    auth_checks = types.ModuleType("litellm.proxy.auth.auth_checks")
    auth_checks.get_key_object = get_key_object

    async def _cache_key_object(**kwargs):
        if cache_key_object is not None:
            await cache_key_object(**kwargs)

    auth_checks._cache_key_object = _cache_key_object
    monkeypatch.setitem(sys.modules, "litellm.proxy.auth.auth_checks", auth_checks)


def _install_fake_spend_callbacks(monkeypatch, captured):
    async def _run_litellm_spend_callbacks(**kwargs):
        captured["spend_callbacks"] = kwargs

    monkeypatch.setattr(
        spend_ingestion_endpoints,
        "_run_litellm_spend_callbacks",
        _run_litellm_spend_callbacks,
    )


def test_should_reject_missing_spend_ingestion_authorization(monkeypatch):
    client = _get_test_client(monkeypatch)

    response = client.post("/internal/spend/ingest", json=_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing LiteLLM master key"


def test_should_reject_invalid_spend_ingestion_master_key(monkeypatch):
    client = _get_test_client(monkeypatch)

    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-not-master"},
        json=_payload(),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid LiteLLM master key"


def test_should_reject_spend_ingestion_when_master_key_is_not_configured(monkeypatch):
    client = _get_test_client(monkeypatch, master_key=None)

    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 500
    assert (
        response.json()["detail"]
        == "LiteLLM master key is required for spend ingestion"
    )


def test_should_register_spend_ingestion_as_master_key_only_route():
    assert "/internal/spend/ingest" in LiteLLMRoutes.master_key_only_routes.value


def test_should_reject_unsupported_call_type(monkeypatch):
    client = _get_test_client(monkeypatch)

    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(call_type="unsupported_call_type"),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == (
        "Unsupported call_type: unsupported_call_type"
    )


def test_should_reject_call_type_outside_spend_ingestion_allowlist(monkeypatch):
    client = _get_test_client(monkeypatch)

    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(call_type="aimage_generation"),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "Unsupported spend ingestion call_type: aimage_generation",
        "supported_call_types": ["_arealtime"],
    }


def test_should_reject_non_finite_spend(monkeypatch):
    client = _get_test_client(monkeypatch)

    payload = _payload()
    payload["spend"] = 0
    body = json.dumps(payload).replace('"spend": 0', '"spend": 1e309')
    response = client.post(
        "/internal/spend/ingest",
        headers={
            "Authorization": "Bearer sk-master",
            "Content-Type": "application/json",
        },
        content=body,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "spend must be a finite non-negative number"


def test_should_return_404_when_api_key_hash_is_missing(monkeypatch):
    from litellm.proxy._types import ProxyErrorTypes, ProxyException

    async def _get_key_object(**kwargs):
        raise ProxyException(
            message="not found",
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=401,
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "api_key_hash was not found"


def test_should_return_500_when_api_key_lookup_fails(monkeypatch):
    async def _get_key_object(**kwargs):
        raise RuntimeError("redis unavailable")

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to lookup api_key_hash"


def test_should_write_generic_spend_payload(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        captured["get_key_object"] = kwargs
        return SimpleNamespace(
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            end_user_id="stale-customer-123",
            key_alias="key-alias-123",
            team_alias="team-alias-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "request_id": "req-123",
        "call_type": "_arealtime",
    }
    assert captured["get_key_object"]["hashed_token"] == "hashed-key"

    spend_callback_call = captured["spend_callbacks"]
    assert spend_callback_call["start_time"].isoformat() == "2026-05-18T01:00:00+00:00"
    assert spend_callback_call["end_time"].isoformat() == "2026-05-18T01:00:03+00:00"

    litellm_kwargs = spend_callback_call["litellm_kwargs"]
    assert litellm_kwargs["call_type"] == "_arealtime"
    assert litellm_kwargs["litellm_call_id"] == "req-123"
    assert litellm_kwargs["litellm_trace_id"] == "sess-123"
    assert litellm_kwargs["response_cost"] == 0.123
    assert litellm_kwargs["litellm_params"]["custom_llm_provider"] is None
    metadata = litellm_kwargs["litellm_params"]["metadata"]
    assert metadata["user_api_key"] == "hashed-key"
    assert metadata["user_api_key_hash"] == "hashed-key"
    assert metadata["user_api_key_alias"] == "key-alias-123"
    assert metadata["user_api_key_user_id"] == "user-123"
    assert metadata["user_api_key_team_id"] == "team-123"
    assert metadata["user_api_key_team_alias"] == "team-alias-123"
    assert metadata["user_api_key_org_id"] == "org-123"
    assert metadata["spend_ingestion"] is True

    response_obj = spend_callback_call["response_obj"]
    assert response_obj["id"] == "req-123"
    assert response_obj["usage"]["total_tokens"] == 30
    assert response_obj["usage"]["input_seconds"] == 1.5

    standard_logging_object = litellm_kwargs["standard_logging_object"]
    assert standard_logging_object["id"] == "req-123"
    assert standard_logging_object["trace_id"] == "sess-123"
    assert standard_logging_object["response_cost"] == 0.123
    assert standard_logging_object["model"] == "gpt-realtime-1.5"
    assert standard_logging_object["model_id"] is None
    assert standard_logging_object["model_group"] == "gpt-realtime-1.5"
    assert standard_logging_object["total_tokens"] == 30
    assert standard_logging_object["prompt_tokens"] == 10
    assert standard_logging_object["completion_tokens"] == 20
    assert standard_logging_object["request_tags"] == []
    assert standard_logging_object["end_user"] == ""
    assert standard_logging_object["metadata"] == metadata


def test_should_use_active_key_hash_for_spend_updates(monkeypatch):
    captured = {}
    user_api_key_auth = SimpleNamespace(
        token="active-key-hash",
        user_id="user-123",
        team_id="team-123",
        org_id="org-123",
    )

    async def _get_key_object(**kwargs):
        captured["get_key_object"] = kwargs
        return user_api_key_auth

    async def _cache_key_object(**kwargs):
        captured["cache_key_object"] = kwargs

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object, _cache_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["api_key_hash"] = "deprecated-key-hash"
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    assert captured["get_key_object"]["hashed_token"] == "deprecated-key-hash"
    assert captured["cache_key_object"]["hashed_token"] == "active-key-hash"
    assert captured["cache_key_object"]["user_api_key_obj"] is user_api_key_auth
    assert (
        captured["cache_key_object"]["user_api_key_cache"]
        is proxy_server.user_api_key_cache
    )
    assert (
        captured["cache_key_object"]["proxy_logging_obj"]
        is proxy_server.proxy_logging_obj
    )

    litellm_kwargs = captured["spend_callbacks"]["litellm_kwargs"]
    metadata = litellm_kwargs["litellm_params"]["metadata"]
    assert metadata["user_api_key"] == "active-key-hash"
    assert metadata["user_api_key_hash"] == "active-key-hash"
    assert (
        litellm_kwargs["standard_logging_object"]["metadata"]["user_api_key_hash"]
        == "active-key-hash"
    )


def test_should_put_usage_object_in_standard_logging_metadata(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["usage"]["cache_read_input_tokens"] = 7
    payload["usage"]["cache_creation_input_tokens"] = 11
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["standard_logging_object"][
        "metadata"
    ]
    assert metadata["usage_object"]["cache_read_input_tokens"] == 7
    assert metadata["usage_object"]["cache_creation_input_tokens"] == 11


def test_should_pass_model_budget_metadata_to_litellm_callbacks(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            model_max_budget={
                "gpt-realtime-1.5": {
                    "max_budget": 1.0,
                    "budget_duration": "1d",
                }
            },
            end_user_model_max_budget={
                "gpt-realtime-1.5": {
                    "max_budget": 0.5,
                    "budget_duration": "1d",
                }
            },
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {"user_api_key_end_user_id": "customer-123"}
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["litellm_params"][
        "metadata"
    ]
    assert metadata["user_api_key_model_max_budget"] == {
        "gpt-realtime-1.5": {
            "max_budget": 1.0,
            "budget_duration": "1d",
        }
    }
    assert metadata["user_api_key_end_user_model_max_budget"] == {
        "gpt-realtime-1.5": {
            "max_budget": 0.5,
            "budget_duration": "1d",
        }
    }
    standard_logging_object = captured["spend_callbacks"]["litellm_kwargs"][
        "standard_logging_object"
    ]
    assert standard_logging_object["metadata"] == metadata
    assert standard_logging_object["end_user"] == "customer-123"


def test_should_pass_router_budget_fields_to_litellm_callbacks(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["custom_llm_provider"] = "openai"
    payload["model_id"] = "deployment-123"
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    litellm_kwargs = captured["spend_callbacks"]["litellm_kwargs"]
    assert litellm_kwargs["custom_llm_provider"] == "openai"
    assert litellm_kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert litellm_kwargs["litellm_params"]["model_info"] == {"id": "deployment-123"}
    metadata = litellm_kwargs["litellm_params"]["metadata"]
    assert metadata["model_info"] == {"id": "deployment-123"}

    standard_logging_object = litellm_kwargs["standard_logging_object"]
    assert standard_logging_object["custom_llm_provider"] == "openai"
    assert standard_logging_object["model_id"] == "deployment-123"
    assert standard_logging_object["hidden_params"]["model_id"] == "deployment-123"


def test_should_use_key_owned_agent_id_for_spend_attribution(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            agent_id="agent-from-key",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {"agent_id": "agent-from-client"}
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["litellm_params"][
        "metadata"
    ]
    assert metadata["agent_id"] == "agent-from-key"
    assert (
        captured["spend_callbacks"]["litellm_kwargs"]["standard_logging_object"][
            "metadata"
        ]["agent_id"]
        == "agent-from-key"
    )


def test_should_preserve_caller_agent_id_when_key_has_none(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {"agent_id": "agent-from-client"}
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["litellm_params"][
        "metadata"
    ]
    assert metadata["agent_id"] == "agent-from-client"


def test_should_skip_spend_updates_when_disabled(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.general_settings = {"disable_spend_updates": True}
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "skipped",
        "reason": "spend updates are disabled",
        "request_id": "req-123",
        "call_type": "_arealtime",
    }
    assert "spend_callbacks" not in captured


def test_should_accept_when_litellm_spend_callback_fails(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    class _FailingProxyDBLogger:
        async def async_log_success_event(self, **kwargs):
            captured["proxy_db_logger"] = kwargs
            raise RuntimeError("db unavailable")

    class _FailingModelMaxBudgetLimiter:
        async def async_log_success_event(self, **kwargs):
            captured["model_max_budget_limiter"] = kwargs
            raise RuntimeError("cache unavailable")

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()
    proxy_server.model_max_budget_limiter = _FailingModelMaxBudgetLimiter()

    proxy_track_cost_callback = types.ModuleType(
        "litellm.proxy.hooks.proxy_track_cost_callback"
    )
    proxy_track_cost_callback._ProxyDBLogger = _FailingProxyDBLogger

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "_get_all_callbacks",
        lambda: [],
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.hooks.proxy_track_cost_callback",
        proxy_track_cost_callback,
    )

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "request_id": "req-123",
        "call_type": "_arealtime",
    }
    assert captured["proxy_db_logger"]["kwargs"]["litellm_call_id"] == "req-123"
    assert (
        captured["model_max_budget_limiter"]["kwargs"]["litellm_call_id"] == "req-123"
    )


def test_should_call_registered_router_budget_callbacks(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    class _ProxyDBLogger:
        async def async_log_success_event(self, **kwargs):
            captured["proxy_db_logger"] = kwargs

    class _ModelMaxBudgetLimiter:
        async def async_log_success_event(self, **kwargs):
            captured["model_max_budget_limiter"] = kwargs

    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

    class _RouterBudgetLimiter(RouterBudgetLimiting):
        async def async_log_success_event(self, **kwargs):
            captured.setdefault("router_budget_limiter", []).append(kwargs)

    router_budget_limiter = _RouterBudgetLimiter.__new__(_RouterBudgetLimiter)
    model_max_budget_limiter = _ModelMaxBudgetLimiter()

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()
    proxy_server.model_max_budget_limiter = model_max_budget_limiter

    proxy_track_cost_callback = types.ModuleType(
        "litellm.proxy.hooks.proxy_track_cost_callback"
    )
    proxy_track_cost_callback._ProxyDBLogger = _ProxyDBLogger

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.hooks.proxy_track_cost_callback",
        proxy_track_cost_callback,
    )
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "_get_all_callbacks",
        lambda: [
            router_budget_limiter,
            router_budget_limiter,
            model_max_budget_limiter,
        ],
    )

    payload = _payload()
    payload["custom_llm_provider"] = "openai"
    payload["model_id"] = "deployment-123"
    payload["metadata"] = {"tags": ["realtime"]}
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    assert captured["proxy_db_logger"]["kwargs"]["litellm_call_id"] == "req-123"
    assert (
        captured["model_max_budget_limiter"]["kwargs"]["litellm_call_id"] == "req-123"
    )
    assert len(captured["router_budget_limiter"]) == 1
    router_budget_kwargs = captured["router_budget_limiter"][0]["kwargs"]
    assert router_budget_kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert router_budget_kwargs["standard_logging_object"]["model_id"] == (
        "deployment-123"
    )
    assert router_budget_kwargs["litellm_params"]["metadata"]["tags"] == ["realtime"]


def test_should_call_registered_session_budget_callbacks(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            agent_id="agent-budget-123",
        )

    class _ProxyDBLogger:
        async def async_log_success_event(self, **kwargs):
            captured["proxy_db_logger"] = kwargs

    class _ModelMaxBudgetLimiter:
        async def async_log_success_event(self, **kwargs):
            captured["model_max_budget_limiter"] = kwargs

    from litellm.proxy.hooks.max_budget_per_session_limiter import (
        _PROXY_MaxBudgetPerSessionHandler,
    )

    class _SessionBudgetLimiter(_PROXY_MaxBudgetPerSessionHandler):
        async def async_log_success_event(self, **kwargs):
            captured.setdefault("session_budget_limiter", []).append(kwargs)

    session_budget_limiter = _SessionBudgetLimiter.__new__(_SessionBudgetLimiter)
    model_max_budget_limiter = _ModelMaxBudgetLimiter()

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()
    proxy_server.model_max_budget_limiter = model_max_budget_limiter

    proxy_track_cost_callback = types.ModuleType(
        "litellm.proxy.hooks.proxy_track_cost_callback"
    )
    proxy_track_cost_callback._ProxyDBLogger = _ProxyDBLogger

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.hooks.proxy_track_cost_callback",
        proxy_track_cost_callback,
    )
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "_get_all_callbacks",
        lambda: [
            session_budget_limiter,
            session_budget_limiter,
            model_max_budget_limiter,
        ],
    )

    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert len(captured["session_budget_limiter"]) == 1
    session_budget_kwargs = captured["session_budget_limiter"][0]["kwargs"]
    metadata = session_budget_kwargs["litellm_params"]["metadata"]
    assert metadata["session_id"] == "sess-123"
    assert metadata["agent_id"] == "agent-budget-123"
    assert session_budget_kwargs["response_cost"] == 0.123
    assert captured["proxy_db_logger"]["kwargs"]["litellm_call_id"] == "req-123"
    assert (
        captured["model_max_budget_limiter"]["kwargs"]["litellm_call_id"] == "req-123"
    )


def test_should_pass_caller_metadata_without_rewriting(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            end_user_id=None,
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {
        "user_api_key_end_user_id": "customer-123",
        "external_request_id": "external-123",
        "tags": ["realtime", "nova"],
        "spend_logs_metadata": {"source": "nova-aigateway"},
    }
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    litellm_kwargs = captured["spend_callbacks"]["litellm_kwargs"]
    metadata = litellm_kwargs["litellm_params"]["metadata"]
    assert metadata["user_api_key_end_user_id"] == "customer-123"
    assert metadata["external_request_id"] == "external-123"
    assert metadata["tags"] == ["realtime", "nova"]
    assert metadata["spend_logs_metadata"] == {"source": "nova-aigateway"}
    standard_logging_object = litellm_kwargs["standard_logging_object"]
    assert standard_logging_object["end_user"] == "customer-123"
    assert standard_logging_object["request_tags"] == ["realtime", "nova"]


def test_should_merge_key_level_spend_metadata(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            token="hashed-key",
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
            metadata={
                "tags": ["key-tag", "shared-tag"],
                "spend_logs_metadata": {
                    "source": "key-config",
                    "shared": "key-value",
                },
            },
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {
        "tags": ["caller-tag", "shared-tag"],
        "spend_logs_metadata": {
            "caller": "payload",
            "shared": "caller-value",
        },
    }
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["litellm_params"][
        "metadata"
    ]
    assert metadata["tags"] == ["caller-tag", "shared-tag", "key-tag"]
    assert metadata["spend_logs_metadata"] == {
        "caller": "payload",
        "shared": "caller-value",
        "source": "key-config",
    }
    assert captured["spend_callbacks"]["litellm_kwargs"]["standard_logging_object"][
        "request_tags"
    ] == ["caller-tag", "shared-tag", "key-tag"]


def test_should_preserve_spend_logs_metadata_without_moving_unknown_keys(monkeypatch):
    captured = {}

    async def _get_key_object(**kwargs):
        return SimpleNamespace(
            user_id="user-123",
            team_id="team-123",
            org_id="org-123",
        )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.prisma_client = _FakePrismaClient()
    proxy_server.user_api_key_cache = SimpleNamespace()
    proxy_server.proxy_logging_obj = SimpleNamespace()

    _install_fake_proxy_server(monkeypatch, proxy_server)
    _install_fake_auth_checks(monkeypatch, _get_key_object)
    _install_fake_spend_callbacks(monkeypatch, captured)

    payload = _payload()
    payload["metadata"] = {
        "external_request_id": "external-123",
        "spend_logs_metadata": {"source": "nova-aigateway"},
    }
    client = _get_test_client(monkeypatch)
    response = client.post(
        "/internal/spend/ingest",
        headers={"Authorization": "Bearer sk-master"},
        json=payload,
    )

    assert response.status_code == 200
    metadata = captured["spend_callbacks"]["litellm_kwargs"]["litellm_params"][
        "metadata"
    ]
    assert metadata["external_request_id"] == "external-123"
    assert metadata["spend_logs_metadata"] == {"source": "nova-aigateway"}
