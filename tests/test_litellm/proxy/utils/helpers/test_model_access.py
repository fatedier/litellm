from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm import ModelResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import (
    create_model_info_response,
    get_available_models_for_user,
    hash_token,
    is_known_model,
    is_known_vector_store_index,
    model_dump_with_preserved_fields,
    validate_model_access,
)


def normalize(value):
    return value


def _router_with_models(model_names):
    router = MagicMock()
    router.get_model_names.return_value = model_names
    router.get_model_access_groups.return_value = {}
    return router


def test_is_known_model_happy_path_returns_true_when_in_router():
    router = _router_with_models(["gpt-4o", "claude-haiku"])
    summary = {
        "result": is_known_model("gpt-4o", router),
        "model": "gpt-4o",
        "router_models": ["gpt-4o", "claude-haiku"],
    }
    assert summary == {
        "result": True,
        "model": "gpt-4o",
        "router_models": ["gpt-4o", "claude-haiku"],
    }


def test_is_known_model_returns_false_when_not_in_router():
    router = _router_with_models(["gpt-4o"])
    summary = {
        "result": is_known_model("claude-haiku", router),
        "model": "claude-haiku",
        "router_models": ["gpt-4o"],
    }
    assert summary == {
        "result": False,
        "model": "claude-haiku",
        "router_models": ["gpt-4o"],
    }


def test_is_known_model_error_path_none_model():
    router = _router_with_models(["gpt-4o"])
    assert is_known_model(None, router) is False


def test_is_known_model_error_path_none_router():
    assert is_known_model("gpt-4o", None) is False


def test_is_known_vector_store_index_happy_path(monkeypatch):
    registry = MagicMock()
    registry.get_vector_store_indexes.return_value = ["index-a", "index-b"]
    monkeypatch.setattr(litellm, "vector_store_index_registry", registry)
    summary = {
        "result": is_known_vector_store_index("index-a"),
        "indexes": ["index-a", "index-b"],
        "input": "index-a",
    }
    assert summary == {
        "result": True,
        "indexes": ["index-a", "index-b"],
        "input": "index-a",
    }


def test_is_known_vector_store_index_returns_false_when_missing(monkeypatch):
    registry = MagicMock()
    registry.get_vector_store_indexes.return_value = ["index-a"]
    monkeypatch.setattr(litellm, "vector_store_index_registry", registry)
    summary = {
        "result": is_known_vector_store_index("missing"),
        "indexes": ["index-a"],
        "input": "missing",
    }
    assert summary == {
        "result": False,
        "indexes": ["index-a"],
        "input": "missing",
    }


def test_is_known_vector_store_index_error_path_no_registry(monkeypatch):
    monkeypatch.setattr(litellm, "vector_store_index_registry", None)
    assert is_known_vector_store_index("anything") is False


def test_create_model_info_response_happy_path_no_metadata():
    result = create_model_info_response(model_id="gpt-4o", provider="openai")
    snapshot = {
        "id": result["id"],
        "object": result["object"],
        "owned_by": result["owned_by"],
        "created_is_int": isinstance(result["created"], int),
        "metadata_absent": "metadata" not in result,
        "max_input_tokens_positive_int": isinstance(result["max_input_tokens"], int)
        and result["max_input_tokens"] > 0,
        "max_output_tokens_positive_int": isinstance(result["max_output_tokens"], int)
        and result["max_output_tokens"] > 0,
    }
    assert snapshot == {
        "id": "gpt-4o",
        "object": "model",
        "owned_by": "openai",
        "created_is_int": True,
        "metadata_absent": True,
        "max_input_tokens_positive_int": True,
        "max_output_tokens_positive_int": True,
    }


def test_create_model_info_response_with_metadata_default_general(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_all_fallbacks",
        lambda **_kwargs: [{"model": "fallback-1"}],
    )
    result = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_metadata=True,
    )
    snapshot = {
        "id": result["id"],
        "owned_by": result["owned_by"],
        "object": result["object"],
        "fallbacks": result["metadata"]["fallbacks"],
    }
    assert snapshot == {
        "id": "gpt-4o",
        "owned_by": "openai",
        "object": "model",
        "fallbacks": [{"model": "fallback-1"}],
    }


def test_create_model_info_response_with_explicit_fallback_type(monkeypatch):
    captured = {}

    def _capture(model, llm_router, fallback_type):
        captured["fallback_type"] = fallback_type
        return ["x"]

    monkeypatch.setattr("litellm.proxy.auth.model_checks.get_all_fallbacks", _capture)
    result = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_metadata=True,
        fallback_type="context_window",
    )
    snapshot = {
        "id": result["id"],
        "fallbacks": result["metadata"]["fallbacks"],
        "captured_fallback_type": captured["fallback_type"],
        "owned_by": result["owned_by"],
    }
    assert snapshot == {
        "id": "gpt-4o",
        "fallbacks": ["x"],
        "captured_fallback_type": "context_window",
        "owned_by": "openai",
    }


def test_create_model_info_response_invalid_fallback_type_raises():
    with pytest.raises(HTTPException) as exc_info:
        create_model_info_response(
            model_id="gpt-4o",
            provider="openai",
            include_metadata=True,
            fallback_type="bogus",
        )
    assert exc_info.value.status_code == 400
    assert "Invalid fallback_type" in str(exc_info.value.detail)


def test_validate_model_access_happy_path_single_model_in_list():
    summary = {
        "result": validate_model_access("gpt-4o", ["gpt-4o", "claude-haiku"]),
        "model": "gpt-4o",
        "available": ["gpt-4o", "claude-haiku"],
    }
    assert summary == {
        "result": None,
        "model": "gpt-4o",
        "available": ["gpt-4o", "claude-haiku"],
    }


def test_validate_model_access_happy_path_batch_all_accessible():
    summary = {
        "result": validate_model_access(
            "gpt-4o,claude-haiku", ["gpt-4o", "claude-haiku", "gemini"]
        ),
        "input": "gpt-4o,claude-haiku",
        "available": ["gpt-4o", "claude-haiku", "gemini"],
    }
    assert summary == {
        "result": None,
        "input": "gpt-4o,claude-haiku",
        "available": ["gpt-4o", "claude-haiku", "gemini"],
    }


def test_validate_model_access_single_model_not_accessible_raises():
    with pytest.raises(HTTPException) as exc_info:
        validate_model_access("missing-model", ["gpt-4o"])
    assert exc_info.value.status_code == 404
    assert "missing-model" in str(exc_info.value.detail)


def test_validate_model_access_batch_partial_inaccessible_raises():
    with pytest.raises(HTTPException) as exc_info:
        validate_model_access("gpt-4o,unknown-x", ["gpt-4o"])
    assert exc_info.value.status_code == 404
    assert "unknown-x" in str(exc_info.value.detail)
    assert "gpt-4o" not in str(exc_info.value.detail).split("not accessible:")[1]


def _make_model_response():
    return ModelResponse(
        id="resp-123",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": "{}"},
                        }
                    ],
                },
                "index": 0,
                "finish_reason": "tool_calls",
            }
        ],
        model="gpt-4o",
    )


def test_model_dump_with_preserved_fields_restores_none_content():
    resp = _make_model_response()
    result = model_dump_with_preserved_fields(resp)
    message = result["choices"][0]["message"]
    snapshot = {
        "content_is_none": message["content"] is None,
        "role": message["role"],
        "has_tool_calls": "tool_calls" in message,
        "model": result["model"],
    }
    assert snapshot == {
        "content_is_none": True,
        "role": "assistant",
        "has_tool_calls": True,
        "model": "gpt-4o",
    }


def test_model_dump_with_preserved_fields_no_choices_returns_plain_dump():
    class _Bare:
        def model_dump(self, **_kwargs):
            return {"id": "x", "object": "y", "extra": "z"}

    bare = _Bare()
    result = model_dump_with_preserved_fields(bare)
    assert result == {"id": "x", "object": "y", "extra": "z"}


def test_model_dump_with_preserved_fields_error_path_invalid_obj_raises():
    with pytest.raises(AttributeError):
        model_dump_with_preserved_fields(None)


@pytest.mark.asyncio
async def test_get_available_models_for_user_happy_path_returns_complete_list(
    monkeypatch,
):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: ["gpt-4o"],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: ["claude-haiku"],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list",
        lambda **_k: ["gpt-4o", "claude-haiku", "gemini"],
    )
    router = _router_with_models(["gpt-4o", "claude-haiku", "gemini"])
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    result = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=router,
        general_settings={},
        user_model=None,
    )
    summary = {
        "result_sorted": sorted(result),
        "count": len(result),
        "user_id": user_api_key_dict.user_id,
        "router_set": True,
    }
    assert summary == {
        "result_sorted": ["claude-haiku", "gemini", "gpt-4o"],
        "count": 3,
        "user_id": "user-1",
        "router_set": True,
    }


@pytest.mark.asyncio
async def test_get_available_models_for_user_with_none_router(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list",
        lambda **_k: ["user-model"],
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    result = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=None,
        general_settings={},
        user_model="user-model",
    )
    summary = {
        "result": result,
        "router_is_none": True,
        "user_model": "user-model",
        "count": len(result),
    }
    assert summary == {
        "result": ["user-model"],
        "router_is_none": True,
        "user_model": "user-model",
        "count": 1,
    }


@pytest.mark.asyncio
async def test_get_available_models_for_user_error_path_complete_list_raises(
    monkeypatch,
):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: [],
    )

    def _boom(**_kwargs):
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list", _boom
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    with pytest.raises(RuntimeError):
        await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
        )


@pytest.mark.asyncio
async def test_get_available_models_for_user_resolves_team_access_group_models(
    monkeypatch,
):
    from litellm.models.access_group import LiteLLM_AccessGroupTable
    from litellm.models.team import LiteLLM_TeamTableCachedObj

    team = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        models=["no-default-models"],
        access_group_ids=["ag-1"],
    )
    access_group = LiteLLM_AccessGroupTable(
        access_group_id="ag-1",
        access_group_name="repro-group",
        access_model_names=["model-a", "model-b"],
        assigned_team_ids=["team-1"],
    )

    async def _get_team_object(**_kwargs):
        return team

    async def _get_access_object(**_kwargs):
        return access_group

    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", _get_team_object)
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_access_object", _get_access_object)

    result = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-1",
            team_id="team-1",
            models=["all-team-models"],
            team_models=["no-default-models"],
        ),
        llm_router=_router_with_models(["model-a", "model-b", "model-c"]),
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )
    assert sorted(result) == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_get_available_models_for_user_without_access_groups_grants_nothing(
    monkeypatch,
):
    from litellm.models.team import LiteLLM_TeamTableCachedObj

    async def _get_team_object(**_kwargs):
        return LiteLLM_TeamTableCachedObj(team_id="team-1", models=["no-default-models"])

    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", _get_team_object)

    result = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-1",
            team_id="team-1",
            models=["all-team-models"],
            team_models=["no-default-models"],
        ),
        llm_router=_router_with_models(["model-a", "model-b"]),
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )
    assert result == []

@pytest.mark.asyncio
async def test_get_available_models_for_user_resolves_key_access_group_models(
    monkeypatch,
):
    from litellm.models.access_group import LiteLLM_AccessGroupTable
    from litellm.models.team import LiteLLM_TeamTableCachedObj

    async def _get_team_object(**_kwargs):
        return LiteLLM_TeamTableCachedObj(team_id="team-1", models=["no-default-models"])

    async def _get_access_object(**_kwargs):
        return LiteLLM_AccessGroupTable(
            access_group_id="ag-1",
            access_group_name="key-group",
            access_model_names=["model-b"],
            assigned_key_ids=[hash_token("sk-test-key")],
        )

    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", _get_team_object)
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_access_object", _get_access_object)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())

    result = await get_available_models_for_user(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-1",
            team_id="team-1",
            models=["no-default-models"],
            team_models=["no-default-models"],
            access_group_ids=["ag-1"],
        ),
        llm_router=_router_with_models(["model-a", "model-b"]),
        general_settings={},
        user_model=None,
        prisma_client=MagicMock(),
        proxy_logging_obj=MagicMock(),
        user_api_key_cache=MagicMock(),
    )
    assert result == ["model-b"]
async def test_filter_models_by_user_access_direct_models():
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.utils import filter_models_by_user_access

    user_object = LiteLLM_UserTable(user_id="u1", models=["m1"])

    result = await filter_models_by_user_access(
        models=["m1", "m2"],
        user_object=user_object,
        llm_router=None,
    )
    assert result == ["m1"]


@pytest.mark.asyncio
async def test_filter_models_by_user_access_includes_access_group_models():
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.utils import filter_models_by_user_access

    user_object = LiteLLM_UserTable(user_id="u1", models=["m1"], access_group_ids=["group-1"])

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        AsyncMock(return_value=["m2"]),
    ):
        result = await filter_models_by_user_access(
            models=["m1", "m2", "m3"],
            user_object=user_object,
            llm_router=None,
        )
    assert result == ["m1", "m2"]


@pytest.mark.asyncio
async def test_filter_models_by_user_access_unrestricted_user_keeps_all():
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.utils import filter_models_by_user_access

    result_none = await filter_models_by_user_access(
        models=["m1", "m2"],
        user_object=None,
        llm_router=None,
    )
    assert result_none == ["m1", "m2"]

    unrestricted = LiteLLM_UserTable(user_id="u1", models=[])
    result_unrestricted = await filter_models_by_user_access(
        models=["m1", "m2"],
        user_object=unrestricted,
        llm_router=None,
    )
    assert result_unrestricted == ["m1", "m2"]


@pytest.mark.asyncio
async def test_get_available_models_for_user_applies_user_object_filter():
    from litellm.proxy._types import LiteLLM_UserTable

    user_object = LiteLLM_UserTable(user_id="u1", models=["m1"])
    key = UserAPIKeyAuth(models=["m1", "m2"], team_models=[], team_id=None)

    all_models = await get_available_models_for_user(
        user_api_key_dict=key,
        llm_router=None,
        general_settings={},
        user_model=None,
        user_object=user_object,
    )
    assert all_models == ["m1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_models, group_models",
    [
        (["no-default-models"], ["m1", "m3"]),
        (["m1"], ["m2"]),
        (["no-default-models"], ["*"]),
        (["no-default-models"], ["all-proxy-models"]),
        (["no-default-models"], ["m-prefix-*"]),
        (["m1"], ["m-prefix-*", "m3"]),
        ([], ["m1"]),
        (["no-default-models"], []),
    ],
)
async def test_filter_models_by_user_access_agrees_with_can_user_call_model(user_models, group_models):
    """The listing filter must decide exactly what request-time authorization
    decides. It is allowed to be faster, but a model it hides has to be one
    can_user_call_model would refuse, and vice versa, or users see models they
    cannot call (or lose models they can)."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy.utils import filter_models_by_user_access

    candidates = ["m1", "m2", "m3", "m-prefix-a", "m-prefix-b", "unrelated"]
    user_object = LiteLLM_UserTable(user_id="u", models=list(user_models), access_group_ids=["g1"])

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        AsyncMock(return_value=list(group_models)),
    ):
        filtered = await filter_models_by_user_access(
            models=list(candidates), user_object=user_object, llm_router=None
        )

        authoritative = []
        for model in candidates:
            try:
                await can_user_call_model(model=model, llm_router=None, user_object=user_object)
                authoritative.append(model)
            except ProxyException:
                pass

    assert filtered == authoritative, f"user_models={user_models} group_models={group_models}"


def _router_with_access_group(group_name: str, group_models: list, alias_map: dict | None = None):
    """Router exposing a config-declared model access group and optional aliases."""
    from collections import defaultdict

    router = MagicMock()

    def _groups(model_name=None, team_id=None):
        groups = defaultdict(list)
        groups[group_name] = list(group_models)
        if model_name is not None and model_name not in group_models:
            return defaultdict(list)
        return groups

    router.get_model_access_groups.side_effect = _groups
    router.model_group_alias = alias_map or {}
    router._get_model_from_alias.side_effect = lambda m: (alias_map or {}).get(m)
    return router


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_models, group_models, alias_map",
    [
        pytest.param(["no-default-models"], ["beta-models"], None, id="group-granted-config-access-group"),
        pytest.param(["no-default-models"], ["a*a*a*b"], None, id="pattern-past-wildcard-bound"),
        pytest.param(["beta-models"], [], None, id="directly-granted-config-access-group"),
        pytest.param(["no-default-models"], ["gpt-4"], {"gpt-4-alias": "gpt-4"}, id="router-alias-of-granted-model"),
        pytest.param(["no-default-models"], ["beta-models", "solo"], None, id="config-access-group-plus-plain-name"),
    ],
)
async def test_filter_models_by_user_access_agrees_with_router_expansions(user_models, group_models, alias_map):
    """The listing filter must also honor what the router expands: a config
    access group named in the allowlist grants every model inside it, and a
    visible alias resolves to its underlying model. Exact-name and wildcard
    matching alone silently drops both."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy.utils import filter_models_by_user_access

    router = _router_with_access_group("beta-models", ["gpt-4", "claude-3"], alias_map)
    candidates = ["gpt-4", "claude-3", "solo", "gpt-4-alias", "unrelated"]
    user_object = LiteLLM_UserTable(user_id="u", models=list(user_models), access_group_ids=["g1"])

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        AsyncMock(return_value=list(group_models)),
    ):
        filtered = await filter_models_by_user_access(
            models=list(candidates), user_object=user_object, llm_router=router
        )
        authoritative = []
        for model in candidates:
            try:
                await can_user_call_model(model=model, llm_router=router, user_object=user_object)
                authoritative.append(model)
            except ProxyException:
                pass

    assert filtered == authoritative, f"user_models={user_models} group_models={group_models} alias={alias_map}"


@pytest.mark.asyncio
async def test_filter_models_by_user_access_agrees_on_global_alias_map():
    """litellm.model_alias_map is a second, router-independent alias source that
    can_user_call_model resolves before matching, so a listed alias whose target
    is allowed must stay listed even with no router configured."""
    from unittest.mock import AsyncMock, patch

    import litellm
    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy.utils import filter_models_by_user_access

    original_alias_map = litellm.model_alias_map
    litellm.model_alias_map = {"gpt-4-alias": "gpt-4"}
    try:
        candidates = ["gpt-4", "gpt-4-alias", "unrelated"]
        user_object = LiteLLM_UserTable(
            user_id="u", models=["no-default-models"], access_group_ids=["g1"]
        )

        with patch(
            "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
            AsyncMock(return_value=["gpt-4"]),
        ):
            filtered = await filter_models_by_user_access(
                models=list(candidates), user_object=user_object, llm_router=None
            )
            authoritative = []
            for model in candidates:
                try:
                    await can_user_call_model(model=model, llm_router=None, user_object=user_object)
                    authoritative.append(model)
                except ProxyException:
                    pass

        assert filtered == authoritative
    finally:
        litellm.model_alias_map = original_alias_map


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group_models, expected",
    [
        pytest.param(["a*"], ["a-normal-model"], id="group-pattern"),
        pytest.param(["*"], ["a-normal-model", "a" * 5000], id="group-grants-everything"),
        pytest.param(
            ["all-proxy-models"], ["a-normal-model", "a" * 5000], id="group-grants-all-proxy-models"
        ),
    ],
)
async def test_filter_models_by_user_access_agrees_on_names_past_the_length_bound(group_models, expected):
    """Authorization refuses to run a wildcard match against a name past the
    length bound, so a catalogue entry that long has to be dropped here too or
    the listing advertises a model every request for it will refuse. A group
    that grants everything has no pattern to run, so the bound must not reach
    it and strip a grant neither path would have spent anything on."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy.utils import filter_models_by_user_access

    candidates = ["a-normal-model", "a" * 5000]
    user_object = LiteLLM_UserTable(
        user_id="u", models=["no-default-models"], access_group_ids=["g1"]
    )

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        AsyncMock(return_value=list(group_models)),
    ):
        filtered = await filter_models_by_user_access(
            models=list(candidates), user_object=user_object, llm_router=None
        )
        authoritative = []
        for model in candidates:
            try:
                await can_user_call_model(model=model, llm_router=None, user_object=user_object)
                authoritative.append(model)
            except ProxyException:
                pass

    assert authoritative == expected
    assert filtered == authoritative


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_models, group_models, candidates",
    [
        pytest.param(["a*a*a*b"], [], ["aaab", "other"], id="direct-pattern-past-bound-no-group"),
        pytest.param(["a*a*a*b"], ["group-model"], ["aaab", "group-model"], id="direct-pattern-past-bound-with-group"),
        pytest.param(["a*a*a*b"], ["c*c*c*d"], ["aaab", "cccd"], id="both-sources-past-bound"),
    ],
)
async def test_filter_models_by_user_access_leaves_direct_grants_unbounded(
    user_models, group_models, candidates
):
    """The bound exists for the grants this feature adds. A user's own models
    are matched unbounded by the direct authorization check, so bounding them
    here would hide models that check still authorizes."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import LiteLLM_UserTable, ProxyException
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy.utils import filter_models_by_user_access

    user_object = LiteLLM_UserTable(
        user_id="u", models=list(user_models), access_group_ids=["g1"] if group_models else []
    )

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        AsyncMock(return_value=list(group_models)),
    ):
        filtered = await filter_models_by_user_access(
            models=list(candidates), user_object=user_object, llm_router=None
        )
        authoritative = []
        for model in candidates:
            try:
                await can_user_call_model(model=model, llm_router=None, user_object=user_object)
                authoritative.append(model)
            except ProxyException:
                pass

    assert filtered == authoritative
