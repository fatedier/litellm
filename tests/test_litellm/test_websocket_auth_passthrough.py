import asyncio
import base64
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import QueryParams
from starlette.websockets import WebSocketState

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.proxy._types import (
    ConfigGeneralSettings,
    LiteLLMRoutes,
    UserAPIKeyAuth,
    WebSocketPassThroughEndpoint,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.pass_through_endpoints import websocket_auth_passthrough
from litellm.proxy.utils import hash_token


def _decode_context(encoded_context):
    padding = "=" * (-len(encoded_context) % 4)
    return json.loads(
        base64.urlsafe_b64decode(encoded_context + padding).decode("utf-8")
    )


class FakeWebSocket:
    def __init__(
        self,
        messages=None,
        headers=None,
        query_string="model=gpt-realtime-1.5&intent=voice",
        path="/v1/realtime",
        root_path="",
    ):
        self._messages = list(messages or [])
        self.headers = headers or {}
        scope_headers = [
            (str(key).lower().encode("utf-8"), str(value).encode("utf-8"))
            for key, value in self.headers.items()
        ]
        self.query_params = QueryParams(query_string)
        self.url = SimpleNamespace(path=path)
        self.scope = {
            "headers": scope_headers,
            "client": ("127.0.0.1", 12345),
            "path": path,
            "query_string": query_string.encode("utf-8"),
            "root_path": root_path,
        }
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self.accept = AsyncMock()
        self.send_text = AsyncMock()
        self.send_bytes = AsyncMock()
        self.close = AsyncMock(side_effect=self._close)

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        return {"type": "websocket.disconnect"}

    def _close(self, *args, **kwargs):
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class FakeUpstream:
    def __init__(self, subprotocol=None):
        self.subprotocol = subprotocol
        self.sent_messages = []
        self.closed = False
        self._client_messages_received = asyncio.Event()

    async def send(self, message):
        self.sent_messages.append(message)
        if len(self.sent_messages) >= 2:
            self._client_messages_received.set()

    async def close(self):
        self.closed = True

    def __aiter__(self):
        self._iter_index = 0
        return self

    async def __anext__(self):
        await self._client_messages_received.wait()
        messages = ["server-text", b"server-bytes"]
        if self._iter_index >= len(messages):
            raise StopAsyncIteration
        message = messages[self._iter_index]
        self._iter_index += 1
        return message


class FakeConnect:
    def __init__(
        self,
        target,
        additional_headers=None,
        subprotocols=None,
        max_size=None,
        selected_subprotocol=None,
    ):
        self.target = target
        self.additional_headers = additional_headers
        self.subprotocols = subprotocols
        self.max_size = max_size
        self.upstream = FakeUpstream(subprotocol=selected_subprotocol)

    async def __aenter__(self):
        return self.upstream

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeApp:
    def __init__(self, routes=None):
        self.routes = list(routes or [])
        self.added_routes = []

    def add_api_websocket_route(self, path, endpoint):
        self.added_routes.append((path, endpoint))
        self.routes.append(SimpleNamespace(path=path, methods=None))


def test_websocket_endpoint_validates_v1_bounds():
    with pytest.raises(ValueError, match="auth=true"):
        WebSocketPassThroughEndpoint(
            path="/v1/realtime",
            target="ws://nova-aigateway/v1/realtime",
            auth=False,
        )

    with pytest.raises(ValueError, match="ws://"):
        WebSocketPassThroughEndpoint(path="/v1/realtime", target="http://example.com")
    with pytest.raises(ValueError, match="ws://"):
        WebSocketPassThroughEndpoint(path="/v1/realtime", target="wss://example.com")


def test_websocket_passthrough_is_not_db_editable_general_setting():
    assert "websocket_pass_through_endpoints" not in ConfigGeneralSettings.model_fields
    endpoint = WebSocketPassThroughEndpoint(
        path="/nova/realtime",
        target="ws://nova-aigateway/v1/realtime",
    )
    assert endpoint.path == "/nova/realtime"


def test_auth_context_uses_inbound_api_key_hash():
    context = websocket_auth_passthrough._build_auth_context(
        api_key_hash=hash_token("sk-master"),
        call_id="call-1",
    )
    encoded = websocket_auth_passthrough._encode_auth_context(context)

    assert _decode_context(encoded) == {
        "schema": "v1",
        "api_key_hash": hash_token("sk-master"),
        "call_id": "call-1",
    }


def test_auth_context_does_not_read_returned_auth_object():
    context = websocket_auth_passthrough._build_auth_context(
        api_key_hash="inbound-key-hash",
        call_id="call-1",
    )

    assert context["api_key_hash"] == "inbound-key-hash"


def test_target_url_merges_query_params():
    endpoint = WebSocketPassThroughEndpoint(
        path="/v1/realtime",
        target="ws://nova-aigateway/v1/realtime?fixed=1",
    )
    websocket = FakeWebSocket(query_string="model=gpt-realtime-1.5&intent=voice")

    assert (
        websocket_auth_passthrough._build_target_url(endpoint, websocket)
        == "ws://nova-aigateway/v1/realtime?fixed=1&model=gpt-realtime-1.5&intent=voice"
    )


def test_secret_subprotocols_are_not_forwarded():
    websocket = FakeWebSocket(
        headers={
            "sec-websocket-protocol": "openai-insecure-api-key.sk-raw, openai-beta.realtime-v1"
        }
    )

    assert websocket_auth_passthrough._get_forwardable_subprotocols(
        websocket=websocket,
        forward_subprotocols=True,
    ) == ["openai-beta.realtime-v1"]


def test_extract_websocket_api_key_prefers_litellm_header():
    websocket = FakeWebSocket(
        headers={
            "authorization": "Bearer sk-auth",
            "x-litellm-api-key": "Bearer sk-litellm",
        }
    )

    assert (
        websocket_auth_passthrough._extract_websocket_api_key(websocket=websocket)
        == "sk-litellm"
    )


def test_extract_websocket_api_key_uses_configured_litellm_header(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(general_settings={"litellm_key_header_name": "x-custom-key"}),
    )
    websocket = FakeWebSocket(
        headers={
            "authorization": "Bearer sk-auth",
            "x-custom-key": "Bearer sk-custom",
        }
    )

    assert (
        websocket_auth_passthrough._extract_websocket_api_key(websocket=websocket)
        == "sk-custom"
    )


def test_extract_websocket_api_key_does_not_fallback_when_custom_header_configured(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(general_settings={"litellm_key_header_name": "x-custom-key"}),
    )
    websocket = FakeWebSocket(headers={"authorization": "Bearer sk-auth"})

    assert (
        websocket_auth_passthrough._extract_websocket_api_key(websocket=websocket)
        is None
    )


def test_registered_websocket_paths_are_llm_api_routes():
    route = "/nova/ws-test"
    original_routes = list(LiteLLMRoutes.openai_routes.value)
    original_llm_routes = list(LiteLLMRoutes.llm_api_routes.value)
    try:
        websocket_auth_passthrough._add_websocket_path_to_llm_routes(route)
        assert route in LiteLLMRoutes.openai_routes.value
        assert route in LiteLLMRoutes.llm_api_routes.value
        assert (
            RouteChecks.is_virtual_key_allowed_to_call_route(
                route=route,
                valid_token=UserAPIKeyAuth(allowed_routes=["llm_api_routes"]),
            )
            is True
        )

        websocket_auth_passthrough._remove_websocket_path_from_llm_routes(route)
        assert route not in LiteLLMRoutes.openai_routes.value
        assert route not in LiteLLMRoutes.llm_api_routes.value
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        LiteLLMRoutes.llm_api_routes.value[:] = original_llm_routes
        websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(
            route
        )


@pytest.mark.asyncio
async def test_initialize_is_idempotent_for_owned_websocket_routes(monkeypatch):
    route = "/nova/realtime"
    fake_app = FakeApp()
    proxy_server_stub = SimpleNamespace(app=fake_app)
    original_routes = list(LiteLLMRoutes.openai_routes.value)
    original_llm_routes = list(LiteLLMRoutes.llm_api_routes.value)

    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)
    websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
        route, None
    )
    websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(route)

    try:
        await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
            [
                WebSocketPassThroughEndpoint(
                    path=route,
                    target="ws://nova-aigateway/v1/realtime",
                )
            ]
        )
        await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
            [
                WebSocketPassThroughEndpoint(
                    path=route,
                    target="ws://nova-aigateway/v2/realtime",
                )
            ]
        )

        assert len(fake_app.added_routes) == 1
        assert (
            websocket_auth_passthrough._registered_websocket_pass_through_routes[
                route
            ].target
            == "ws://nova-aigateway/v2/realtime"
        )
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        LiteLLMRoutes.llm_api_routes.value[:] = original_llm_routes
        websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
            route, None
        )
        websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(
            route
        )


@pytest.mark.asyncio
async def test_initialize_does_not_remove_omitted_websocket_routes(monkeypatch):
    route = "/nova/realtime"
    fake_app = FakeApp()
    proxy_server_stub = SimpleNamespace(app=fake_app)
    original_routes = list(LiteLLMRoutes.openai_routes.value)
    original_llm_routes = list(LiteLLMRoutes.llm_api_routes.value)

    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)
    websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
        route, None
    )
    websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(route)

    try:
        await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
            [
                WebSocketPassThroughEndpoint(
                    path=route,
                    target="ws://nova-aigateway/v1/realtime",
                )
            ]
        )
        await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
            []
        )

        assert (
            route
            in websocket_auth_passthrough._registered_websocket_pass_through_routes
        )
        assert len(fake_app.added_routes) == 1
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        LiteLLMRoutes.llm_api_routes.value[:] = original_llm_routes
        websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
            route, None
        )
        websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(
            route
        )


@pytest.mark.asyncio
async def test_initialize_rejects_existing_websocket_route_collision(monkeypatch):
    route = "/v1/realtime"
    fake_app = FakeApp(routes=[SimpleNamespace(path=route, methods=None)])
    original_routes = list(LiteLLMRoutes.openai_routes.value)
    proxy_server_stub = SimpleNamespace(app=fake_app)

    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)
    websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
        route, None
    )
    websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(route)

    try:
        with pytest.raises(ValueError, match="collides with an existing app route"):
            await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
                [
                    WebSocketPassThroughEndpoint(
                        path=route,
                        target="ws://nova-aigateway/v1/realtime",
                    )
                ]
            )

        assert (
            route
            not in websocket_auth_passthrough._registered_websocket_pass_through_routes
        )
        assert fake_app.added_routes == []
        assert LiteLLMRoutes.openai_routes.value == original_routes
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
            route, None
        )
        websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(
            route
        )


@pytest.mark.asyncio
async def test_initialize_rejects_existing_http_route_collision(monkeypatch):
    route = "/key/generate"
    fake_app = FakeApp(routes=[SimpleNamespace(path=route, methods={"POST"})])
    original_routes = list(LiteLLMRoutes.openai_routes.value)
    original_llm_routes = list(LiteLLMRoutes.llm_api_routes.value)
    proxy_server_stub = SimpleNamespace(app=fake_app)

    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_stub)
    websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
        route, None
    )
    websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(route)

    try:
        with pytest.raises(ValueError, match="collides with an existing app route"):
            await websocket_auth_passthrough.initialize_websocket_auth_passthrough_endpoints(
                [
                    WebSocketPassThroughEndpoint(
                        path=route,
                        target="ws://nova-aigateway/v1/realtime",
                    )
                ]
            )

        assert (
            route
            not in websocket_auth_passthrough._registered_websocket_pass_through_routes
        )
        assert fake_app.added_routes == []
        assert LiteLLMRoutes.openai_routes.value == original_routes
        assert LiteLLMRoutes.llm_api_routes.value == original_llm_routes
    finally:
        LiteLLMRoutes.openai_routes.value[:] = original_routes
        LiteLLMRoutes.llm_api_routes.value[:] = original_llm_routes
        websocket_auth_passthrough._registered_websocket_pass_through_routes.pop(
            route, None
        )
        websocket_auth_passthrough._websocket_passthrough_llm_routes_added.discard(
            route
        )


@pytest.mark.asyncio
async def test_passthrough_auth_does_not_synthesize_model_query(monkeypatch):
    captured_auth = {}

    async def _fake_user_api_key_auth(request, api_key):
        captured_auth["api_key"] = api_key
        captured_auth["route"] = request.url.path
        captured_auth["method"] = request.method
        captured_auth["body"] = await request.body()
        captured_auth["client"] = request.client.host
        captured_auth["query_params"] = dict(request.query_params)
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        websocket_auth_passthrough,
        "user_api_key_auth",
        _fake_user_api_key_auth,
    )
    websocket = FakeWebSocket(
        headers={"authorization": "Bearer sk-raw"},
        query_string="model=upstream-only-model",
        path="/nova/realtime",
    )

    api_key_hash = await websocket_auth_passthrough.websocket_auth_passthrough_key_auth(
        websocket
    )

    assert api_key_hash == hash_token("sk-raw")
    assert captured_auth == {
        "api_key": "Bearer sk-raw",
        "route": "/nova/realtime",
        "method": "GET",
        "body": b"{}",
        "client": "127.0.0.1",
        "query_params": {},
    }
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_passthrough_auth_rejects_non_sk_api_keys(monkeypatch):
    user_api_key_auth = AsyncMock()
    monkeypatch.setattr(
        websocket_auth_passthrough,
        "user_api_key_auth",
        user_api_key_auth,
    )
    websocket = FakeWebSocket(headers={"authorization": "Bearer opaque-token"})

    with pytest.raises(HTTPException) as exc_info:
        await websocket_auth_passthrough.websocket_auth_passthrough_key_auth(websocket)

    assert exc_info.value.status_code == 403
    assert "sk-" in exc_info.value.detail
    websocket.close.assert_awaited_once_with(code=1008)
    user_api_key_auth.assert_not_awaited()


@pytest.mark.asyncio
async def test_passthrough_auth_strips_model_query_only(monkeypatch):
    captured_auth = {}

    async def _fake_user_api_key_auth(request, api_key):
        captured_auth["query_params"] = dict(request.query_params)
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        websocket_auth_passthrough,
        "user_api_key_auth",
        _fake_user_api_key_auth,
    )
    websocket = FakeWebSocket(
        headers={"authorization": "Bearer sk-raw"},
        query_string="model=upstream-only-model&session_id=session-1",
    )

    await websocket_auth_passthrough.websocket_auth_passthrough_key_auth(websocket)

    assert captured_auth["query_params"] == {
        "session_id": "session-1",
    }


@pytest.mark.asyncio
async def test_passthrough_auth_preserves_root_path_for_route_checks(monkeypatch):
    captured_auth = {}

    async def _fake_user_api_key_auth(request, api_key):
        captured_auth["route"] = request.url.path[len(request.base_url.path) - 1 :]
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        websocket_auth_passthrough,
        "user_api_key_auth",
        _fake_user_api_key_auth,
    )
    websocket = FakeWebSocket(
        headers={"authorization": "Bearer sk-raw"},
        path="/litellm/nova/realtime",
        root_path="/litellm",
    )

    await websocket_auth_passthrough.websocket_auth_passthrough_key_auth(websocket)

    assert captured_auth["route"] == "/nova/realtime"


@pytest.mark.asyncio
async def test_close_uses_application_state_to_avoid_double_close():
    websocket = FakeWebSocket()
    websocket.client_state = WebSocketState.CONNECTED
    websocket.application_state = WebSocketState.DISCONNECTED

    await websocket_auth_passthrough._close_websocket_if_connected(websocket)

    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_websocket_auth_passthrough_forwards_frames_and_internal_headers(
    monkeypatch,
):
    fake_connects = []

    def _fake_connect(*args, **kwargs):
        fake_connect = FakeConnect(*args, **kwargs, selected_subprotocol="proto-b")
        fake_connects.append(fake_connect)
        return fake_connect

    monkeypatch.setattr(websocket_auth_passthrough, "connect", _fake_connect)
    monkeypatch.setattr(
        websocket_auth_passthrough.uuid,
        "uuid4",
        lambda: "call-123",
    )

    websocket = FakeWebSocket(
        messages=[
            {"type": "websocket.receive", "text": "client-text"},
            {"type": "websocket.receive", "bytes": b"client-bytes"},
            {"type": "websocket.disconnect"},
        ],
        headers={
            "authorization": "Bearer sk-raw",
            "sec-websocket-protocol": "openai-insecure-api-key.sk-raw, proto-a, proto-b",
            "X-Api-Resource-Id": "client-resource",
        },
    )
    endpoint = WebSocketPassThroughEndpoint(
        path="/v1/realtime",
        target="ws://nova-aigateway/v1/realtime",
        headers={"Authorization": "Bearer service-token"},
    )

    await websocket_auth_passthrough.websocket_auth_passthrough_request(
        websocket=websocket,
        endpoint=endpoint,
        api_key_hash="hashed-key",
    )

    assert len(fake_connects) == 1
    fake_connect = fake_connects[0]
    assert (
        fake_connect.target
        == "ws://nova-aigateway/v1/realtime?model=gpt-realtime-1.5&intent=voice"
    )
    assert fake_connect.subprotocols == ["proto-a", "proto-b"]
    assert fake_connect.max_size == REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    assert fake_connect.additional_headers["Authorization"] == "Bearer service-token"
    assert "X-Api-Resource-Id" not in fake_connect.additional_headers
    assert "sk-raw" not in json.dumps(fake_connect.additional_headers)
    assert _decode_context(
        fake_connect.additional_headers[
            websocket_auth_passthrough.LITELLM_AUTH_CONTEXT_HEADER_NAME
        ]
    ) == {
        "schema": "v1",
        "api_key_hash": "hashed-key",
        "call_id": "call-123",
    }
    websocket.accept.assert_awaited_once_with(subprotocol="proto-b")
    assert fake_connect.upstream.sent_messages == ["client-text", b"client-bytes"]
    websocket.send_text.assert_awaited_once_with("server-text")
    websocket.send_bytes.assert_awaited_once_with(b"server-bytes")


@pytest.mark.asyncio
async def test_websocket_auth_passthrough_forwards_safe_client_headers_when_enabled(
    monkeypatch,
):
    fake_connects = []

    def _fake_connect(*args, **kwargs):
        fake_connect = FakeConnect(*args, **kwargs)
        fake_connects.append(fake_connect)
        return fake_connect

    monkeypatch.setattr(websocket_auth_passthrough, "connect", _fake_connect)
    monkeypatch.setattr(
        websocket_auth_passthrough.uuid,
        "uuid4",
        lambda: "call-123",
    )

    websocket = FakeWebSocket(
        messages=[{"type": "websocket.disconnect"}],
        headers={
            "Authorization": "Bearer sk-raw",
            "Cookie": "session=raw",
            "Host": "litellm.example.com",
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Key": "raw-key",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Protocol": "openai-insecure-api-key.sk-raw, proto-a",
            "Proxy-Authorization": "Basic raw",
            "Content-Length": "123",
            "Transfer-Encoding": "chunked",
            "X-LiteLLM-Api-Key": "sk-raw",
            "X-LiteLLM-Auth-Context": "raw-context",
            "api-key": "sk-raw",
            "X-Api-Key": "client-upstream-key",
            "X-Api-Resource-Id": "volc.seedasr.sauc.duration",
            "X-Api-Request-Id": "request-1",
            "X-Api-Connect-Id": "connect-1",
            "X-Api-Sequence": "-1",
            "X-Custom-Business": "business",
        },
    )
    endpoint = WebSocketPassThroughEndpoint(
        path="/volcengine/api/v3/sauc/bigmodel_async",
        target="ws://nova-aigateway/volcengine/api/v3/sauc/bigmodel_async",
        headers={"x-api-key": "managed-upstream-key"},
        forward_headers=True,
    )

    await websocket_auth_passthrough.websocket_auth_passthrough_request(
        websocket=websocket,
        endpoint=endpoint,
        api_key_hash="hashed-key",
    )

    assert len(fake_connects) == 1
    upstream_headers = fake_connects[0].additional_headers
    assert upstream_headers["x-api-key"] == "managed-upstream-key"
    assert upstream_headers["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
    assert upstream_headers["X-Api-Request-Id"] == "request-1"
    assert upstream_headers["X-Api-Connect-Id"] == "connect-1"
    assert upstream_headers["X-Api-Sequence"] == "-1"
    assert upstream_headers["X-Custom-Business"] == "business"
    assert "Authorization" not in upstream_headers
    assert "Cookie" not in upstream_headers
    assert "Host" not in upstream_headers
    assert "Connection" not in upstream_headers
    assert "Upgrade" not in upstream_headers
    assert "Sec-WebSocket-Key" not in upstream_headers
    assert "Sec-WebSocket-Version" not in upstream_headers
    assert "Sec-WebSocket-Protocol" not in upstream_headers
    assert "Proxy-Authorization" not in upstream_headers
    assert "Content-Length" not in upstream_headers
    assert "Transfer-Encoding" not in upstream_headers
    assert "X-LiteLLM-Api-Key" not in upstream_headers
    assert "api-key" not in upstream_headers
    assert (
        upstream_headers[websocket_auth_passthrough.LITELLM_AUTH_CONTEXT_HEADER_NAME]
        != "raw-context"
    )
    assert _decode_context(
        upstream_headers[websocket_auth_passthrough.LITELLM_AUTH_CONTEXT_HEADER_NAME]
    ) == {
        "schema": "v1",
        "api_key_hash": "hashed-key",
        "call_id": "call-123",
    }


@pytest.mark.asyncio
async def test_websocket_auth_passthrough_accepts_without_subprotocol_when_upstream_declines(
    monkeypatch,
):
    def _fake_connect(*args, **kwargs):
        return FakeConnect(*args, **kwargs, selected_subprotocol=None)

    monkeypatch.setattr(websocket_auth_passthrough, "connect", _fake_connect)

    websocket = FakeWebSocket(
        messages=[{"type": "websocket.disconnect"}],
        headers={"sec-websocket-protocol": "proto-a, proto-b"},
    )
    endpoint = WebSocketPassThroughEndpoint(
        path="/v1/realtime",
        target="ws://nova-aigateway/v1/realtime",
    )

    await websocket_auth_passthrough.websocket_auth_passthrough_request(
        websocket=websocket,
        endpoint=endpoint,
        api_key_hash="hashed-key",
    )

    websocket.accept.assert_awaited_once_with()
