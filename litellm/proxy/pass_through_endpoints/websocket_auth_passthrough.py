import asyncio
import base64
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, status
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.proxy._types import (
    LiteLLMRoutes,
    SpecialHeaders,
    WebSocketPassThroughEndpoint,
)
from litellm.proxy.auth.user_api_key_auth import (
    _get_bearer_token,
    _get_bearer_token_or_received_api_key,
    user_api_key_auth,
)
from litellm.proxy.utils import hash_token
from litellm.proxy.utils import normalize_route_for_root_path

from .pass_through_endpoints import set_env_variables_in_header

LITELLM_AUTH_CONTEXT_HEADER_NAME = "X-LiteLLM-Auth-Context"
OPENAI_INSECURE_API_KEY_SUBPROTOCOL_PREFIX = "openai-insecure-api-key."
WEBSOCKET_PASSTHROUGH_DENYLISTED_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "cookie",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "api-key",
}
WEBSOCKET_PASSTHROUGH_DENYLISTED_HEADER_PREFIXES = (
    "proxy-",
    "sec-websocket-",
    "x-litellm-",
)

_registered_websocket_pass_through_routes: Dict[str, WebSocketPassThroughEndpoint] = {}
_websocket_passthrough_llm_routes_added: set[str] = set()


def _normalize_websocket_path(path: str) -> str:
    normalized_path = normalize_route_for_root_path(path)
    if normalized_path is not None:
        return normalized_path
    return path


def get_websocket_auth_passthrough_endpoint(
    path: str,
) -> Optional[WebSocketPassThroughEndpoint]:
    return _registered_websocket_pass_through_routes.get(
        _normalize_websocket_path(path)
    )


def _is_app_route_registered(app: FastAPI, path: str) -> bool:
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return True
    return False


def _ensure_websocket_path_is_available(app: FastAPI, path: str) -> None:
    if _is_app_route_registered(app=app, path=path):
        raise ValueError(
            f"websocket passthrough path {path} collides with an existing app route"
        )


def _coerce_websocket_passthrough_endpoint(
    endpoint: Union[Dict[str, Any], WebSocketPassThroughEndpoint],
) -> WebSocketPassThroughEndpoint:
    if isinstance(endpoint, WebSocketPassThroughEndpoint):
        return endpoint
    return WebSocketPassThroughEndpoint(**endpoint)


async def initialize_websocket_auth_passthrough_endpoints(
    websocket_pass_through_endpoints: Sequence[
        Union[Dict[str, Any], WebSocketPassThroughEndpoint]
    ],
) -> None:
    from litellm.proxy.proxy_server import app

    for raw_endpoint in websocket_pass_through_endpoints:
        endpoint = _coerce_websocket_passthrough_endpoint(raw_endpoint)
        normalized_path = _normalize_websocket_path(endpoint.path)
        if normalized_path in _registered_websocket_pass_through_routes:
            _registered_websocket_pass_through_routes[normalized_path] = endpoint
            continue
        _ensure_websocket_path_is_available(app=app, path=normalized_path)
        _registered_websocket_pass_through_routes[normalized_path] = endpoint
        _add_websocket_path_to_llm_routes(normalized_path)
        app.add_api_websocket_route(
            path=normalized_path,
            endpoint=create_websocket_auth_passthrough_route(normalized_path),
        )


def _add_websocket_path_to_llm_routes(path: str) -> None:
    if path not in LiteLLMRoutes.openai_routes.value:
        LiteLLMRoutes.openai_routes.value.append(path)
    if path not in LiteLLMRoutes.llm_api_routes.value:
        LiteLLMRoutes.llm_api_routes.value.append(path)
    _websocket_passthrough_llm_routes_added.add(path)


def _remove_websocket_path_from_llm_routes(path: str) -> None:
    if path not in _websocket_passthrough_llm_routes_added:
        return
    _websocket_passthrough_llm_routes_added.discard(path)
    if path in LiteLLMRoutes.openai_routes.value:
        LiteLLMRoutes.openai_routes.value.remove(path)
    if path in LiteLLMRoutes.llm_api_routes.value:
        LiteLLMRoutes.llm_api_routes.value.remove(path)


def create_websocket_auth_passthrough_route(path: str):
    async def websocket_auth_passthrough_endpoint(
        websocket: WebSocket,
        api_key_hash: str = Depends(websocket_auth_passthrough_key_auth),
    ):
        # This passthrough is intentionally auth-only. LiteLLM authenticates the
        # caller and forwards the verified identity to the upstream; model-level
        # authorization for the WebSocket protocol is owned by the upstream.
        endpoint = get_websocket_auth_passthrough_endpoint(path)
        if endpoint is None:
            await websocket.close(
                code=1008, reason="WebSocket passthrough not configured"
            )
            return
        return await websocket_auth_passthrough_request(
            websocket=websocket,
            endpoint=endpoint,
            api_key_hash=api_key_hash,
        )

    return websocket_auth_passthrough_endpoint


async def websocket_auth_passthrough_key_auth(websocket: WebSocket) -> str:
    websocket_scope = getattr(websocket, "scope", {}) or {}
    scope_headers = list(websocket_scope.get("headers") or [])
    query_string = _strip_model_from_query_string(
        websocket_scope.get("query_string", b"")
    )
    request_scope = {
        "type": "http",
        "method": "GET",
        "headers": scope_headers,
        "query_string": query_string,
    }
    if "client" in websocket_scope:
        request_scope["client"] = websocket_scope["client"]
    for scope_key in ("path", "root_path"):
        if scope_key in websocket_scope:
            request_scope[scope_key] = websocket_scope[scope_key]
    request = Request(scope=request_scope)
    request._url = websocket.url

    async def return_empty_body() -> bytes:
        # This passthrough is auth-only. Do not synthesize realtime
        # {"model": "..."} from the query string, or standard auth will run
        # LiteLLM model allowlist checks before the upstream sees the request.
        return b"{}"

    request.body = return_empty_body  # type: ignore

    api_key = _extract_websocket_api_key(websocket=websocket)
    if not api_key:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail="No API key provided")
    if not api_key.startswith("sk-"):
        # WebSocket auth passthrough v1 only supports LiteLLM native virtual/master
        # keys. OAuth2/JWT/custom opaque credentials are intentionally rejected so
        # X-LiteLLM-Auth-Context never forwards raw bearer material upstream.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=403,
            detail="WebSocket passthrough only supports LiteLLM sk- API keys",
        )

    # The upstream identity must be the inbound LiteLLM key hash, not fields
    # returned by custom auth or other auth hooks.
    api_key_hash = hash_token(api_key)

    try:
        await user_api_key_auth(request=request, api_key=f"Bearer {api_key}")
        return api_key_hash
    except Exception as e:
        verbose_proxy_logger.exception(e)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail=str(e))


def _get_websocket_header(websocket: WebSocket, header_name: str) -> Optional[str]:
    headers = websocket.headers
    header_value = headers.get(header_name)
    if header_value is not None:
        return header_value

    lower_header_name = header_name.lower()
    if hasattr(headers, "items"):
        for key, value in headers.items():
            if str(key).lower() == lower_header_name:
                return str(value)
    return None


def _strip_model_from_query_string(query_string: Union[bytes, str]) -> bytes:
    raw_query_string = (
        query_string.encode("utf-8") if isinstance(query_string, str) else query_string
    )
    query_pairs = parse_qsl(raw_query_string.decode("utf-8"), keep_blank_values=True)
    filtered_query_pairs = [(k, v) for k, v in query_pairs if k != "model"]
    return urlencode(filtered_query_pairs).encode("utf-8")


def _get_configured_litellm_key_header_name() -> Optional[str]:
    proxy_server_module = sys.modules.get("litellm.proxy.proxy_server")
    general_settings = getattr(proxy_server_module, "general_settings", None)
    if not isinstance(general_settings, dict):
        return None
    configured_header_name = general_settings.get("litellm_key_header_name")
    if not isinstance(configured_header_name, str) or not configured_header_name:
        return None
    return configured_header_name


def _extract_websocket_api_key(websocket: WebSocket) -> Optional[str]:
    configured_header_name = _get_configured_litellm_key_header_name()
    if configured_header_name is not None:
        # Match HTTP auth semantics: once litellm_key_header_name is configured,
        # that header is the only LiteLLM credential source for this tunnel.
        configured_api_key = _get_websocket_header(
            websocket=websocket,
            header_name=configured_header_name,
        )
        return _get_bearer_token(configured_api_key) if configured_api_key else None

    litellm_api_key = _get_websocket_header(
        websocket=websocket,
        header_name=SpecialHeaders.custom_litellm_api_key.value,
    )
    if litellm_api_key:
        return _get_bearer_token_or_received_api_key(litellm_api_key)

    authorization = _get_websocket_header(
        websocket=websocket, header_name="authorization"
    )
    if authorization:
        if not authorization.startswith("Bearer "):
            return None
        return authorization[len("Bearer ") :].strip()

    api_key = _get_websocket_header(websocket=websocket, header_name="api-key")
    if api_key:
        return api_key

    for protocol in (
        _get_websocket_header(websocket, "sec-websocket-protocol") or ""
    ).split(","):
        protocol = protocol.strip()
        if protocol.startswith(OPENAI_INSECURE_API_KEY_SUBPROTOCOL_PREFIX):
            return protocol[len(OPENAI_INSECURE_API_KEY_SUBPROTOCOL_PREFIX) :]

    return None


def _get_query_param_pairs(websocket: WebSocket) -> List[Tuple[str, str]]:
    query_params = getattr(websocket, "query_params", None)
    if query_params is None:
        return []
    if hasattr(query_params, "multi_items"):
        return [(str(k), str(v)) for k, v in query_params.multi_items()]
    return [(str(k), str(v)) for k, v in dict(query_params).items()]


def _build_target_url(
    endpoint: WebSocketPassThroughEndpoint, websocket: WebSocket
) -> str:
    if endpoint.forward_query_params is not True:
        return endpoint.target

    split_target = urlsplit(endpoint.target)
    query_pairs = parse_qsl(split_target.query, keep_blank_values=True)
    query_pairs.extend(_get_query_param_pairs(websocket))
    return urlunsplit(
        (
            split_target.scheme,
            split_target.netloc,
            split_target.path,
            urlencode(query_pairs),
            split_target.fragment,
        )
    )


def _get_requested_subprotocols(websocket: WebSocket) -> List[str]:
    requested_protocols = websocket.headers.get("sec-websocket-protocol") or ""
    return [
        protocol.strip()
        for protocol in requested_protocols.split(",")
        if protocol.strip()
    ]


def _is_secret_subprotocol(protocol: str) -> bool:
    return protocol.startswith(OPENAI_INSECURE_API_KEY_SUBPROTOCOL_PREFIX)


def _get_forwardable_subprotocols(
    websocket: WebSocket,
    forward_subprotocols: bool,
) -> List[str]:
    if forward_subprotocols is not True:
        return []
    return [
        protocol
        for protocol in _get_requested_subprotocols(websocket)
        if not _is_secret_subprotocol(protocol)
    ]


def _build_auth_context(api_key_hash: str, call_id: str) -> Dict[str, str]:
    return {
        "schema": "v1",
        "api_key_hash": api_key_hash,
        "call_id": call_id,
    }


def _encode_auth_context(context: Dict[str, str]) -> str:
    raw_context = json.dumps(context, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw_context).decode("ascii").rstrip("=")


def _set_header_case_insensitive(
    headers: Dict[str, str], header_name: str, header_value: str
) -> None:
    lower_header_name = header_name.lower()
    for existing_header in list(headers):
        if existing_header.lower() == lower_header_name:
            del headers[existing_header]
    headers[header_name] = header_value


def _is_forwardable_client_header(header_name: str) -> bool:
    lower_header_name = header_name.lower()
    configured_header_name = _get_configured_litellm_key_header_name()
    if (
        configured_header_name is not None
        and lower_header_name == configured_header_name.lower()
    ):
        return False
    if lower_header_name in WEBSOCKET_PASSTHROUGH_DENYLISTED_HEADERS:
        return False
    return not lower_header_name.startswith(
        WEBSOCKET_PASSTHROUGH_DENYLISTED_HEADER_PREFIXES
    )


def _forwardable_client_headers(websocket: WebSocket) -> Dict[str, str]:
    headers = getattr(websocket, "headers", {})
    if not hasattr(headers, "items"):
        return {}

    upstream_headers: Dict[str, str] = {}
    for header_name, header_value in headers.items():
        header_name = str(header_name).strip()
        if not header_name or not _is_forwardable_client_header(header_name):
            continue
        _set_header_case_insensitive(
            headers=upstream_headers,
            header_name=header_name,
            header_value=str(header_value),
        )
    return upstream_headers


async def _build_upstream_headers(
    websocket: WebSocket,
    endpoint: WebSocketPassThroughEndpoint,
    api_key_hash: str,
    call_id: str,
) -> Dict[str, str]:
    upstream_headers = (
        _forwardable_client_headers(websocket=websocket)
        if endpoint.forward_headers is True
        else {}
    )
    headers = await set_env_variables_in_header(dict(endpoint.headers or {}))
    for header_name, header_value in (headers or {}).items():
        _set_header_case_insensitive(
            headers=upstream_headers,
            header_name=str(header_name),
            header_value=str(header_value),
        )
    if endpoint.inject_litellm_auth_context is True:
        _set_header_case_insensitive(
            headers=upstream_headers,
            header_name=LITELLM_AUTH_CONTEXT_HEADER_NAME,
            header_value=_encode_auth_context(
                _build_auth_context(api_key_hash=api_key_hash, call_id=call_id)
            ),
        )
    return upstream_headers


async def _forward_client_to_upstream(websocket: WebSocket, upstream_ws: Any) -> None:
    while True:
        message = await websocket.receive()
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            await upstream_ws.close()
            return

        text_data = message.get("text")
        bytes_data = message.get("bytes")
        if text_data is not None:
            await upstream_ws.send(text_data)
        elif bytes_data is not None:
            await upstream_ws.send(bytes_data)


async def _forward_upstream_to_client(websocket: WebSocket, upstream_ws: Any) -> None:
    async for upstream_message in upstream_ws:
        if isinstance(upstream_message, bytes):
            await websocket.send_bytes(upstream_message)
        else:
            await websocket.send_text(str(upstream_message))


async def _run_websocket_pipe(websocket: WebSocket, upstream_ws: Any) -> None:
    tasks = [
        asyncio.create_task(_forward_client_to_upstream(websocket, upstream_ws)),
        asyncio.create_task(_forward_upstream_to_client(websocket, upstream_ws)),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    for task in done:
        exception = task.exception()
        if exception is not None:
            raise exception


async def _accept_websocket(
    websocket: WebSocket, subprotocol: Optional[str] = None
) -> None:
    if subprotocol is not None:
        await websocket.accept(subprotocol=subprotocol)
    else:
        await websocket.accept()


async def _close_websocket_if_connected(
    websocket: WebSocket,
    code: int = 1000,
    reason: Optional[str] = None,
) -> None:
    if (
        websocket.application_state != WebSocketState.DISCONNECTED
        and websocket.client_state != WebSocketState.DISCONNECTED
    ):
        await websocket.close(code=code, reason=reason)


async def websocket_auth_passthrough_request(
    websocket: WebSocket,
    endpoint: WebSocketPassThroughEndpoint,
    api_key_hash: str,
) -> None:
    call_id = str(uuid.uuid4())
    target = _build_target_url(endpoint=endpoint, websocket=websocket)
    subprotocols = _get_forwardable_subprotocols(
        websocket=websocket,
        forward_subprotocols=endpoint.forward_subprotocols,
    )
    upstream_headers = await _build_upstream_headers(
        websocket=websocket,
        endpoint=endpoint,
        api_key_hash=api_key_hash,
        call_id=call_id,
    )

    # This route is intentionally a thin auth tunnel. It does not run LiteLLM
    # pre-call hooks, guardrails, RPM/TPM/max-parallel callbacks, or spend
    # logging. Nova owns realtime protocol policy, throttling, and billing for
    # traffic forwarded through this path.
    try:
        async with connect(
            target,
            additional_headers=upstream_headers,
            subprotocols=subprotocols or None,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
        ) as upstream_ws:
            await _accept_websocket(
                websocket=websocket,
                subprotocol=getattr(upstream_ws, "subprotocol", None),
            )
            await _run_websocket_pipe(websocket=websocket, upstream_ws=upstream_ws)
    except InvalidStatus:
        verbose_proxy_logger.exception(
            "WebSocket auth passthrough upstream rejected connection"
        )
        await _close_websocket_if_connected(
            websocket=websocket,
            code=1011,
            reason="Upstream connection rejected",
        )
    except Exception:
        verbose_proxy_logger.exception("WebSocket auth passthrough failed")
        await _close_websocket_if_connected(
            websocket=websocket,
            code=1011,
            reason="WebSocket passthrough error",
        )
    finally:
        await _close_websocket_if_connected(websocket=websocket)
