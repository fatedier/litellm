import asyncio
import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, Mock

import httpx
import litellm
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from litellm.integrations.nova_aigateway import (
    _PublicKeyRecord,
    NovaAIGatewayLogService,
)
from litellm.litellm_core_utils import litellm_logging


def _build_public_key_record(
    key_id: str,
    pem: str,
    version: str = "projects/test/locations/global/keyRings/ring/cryptoKeys/alice/cryptoKeyVersions/1",
) -> _PublicKeyRecord:
    return _PublicKeyRecord(
        key_id=key_id,
        pem=pem,
        version=version,
        algorithm="RSA_DECRYPT_OAEP_2048_SHA256",
        public_key=serialization.load_pem_public_key(pem.encode("utf-8")),
    )


def _standard_logging_object(
    user_id: str = "alice",
    start_time: Optional[float] = None,
) -> dict:
    if start_time is None:
        start_time = datetime(2026, 4, 21, 16, 35, 12, tzinfo=timezone.utc).timestamp()
    return {
        "id": "req-1",
        "startTime": start_time,
        "metadata": {"user_api_key_user_id": user_id},
        "model": "openai/gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "response": {"id": "resp-1"},
        "status": "success",
        "response_time": 1.23,
    }


def test_nova_aigateway_normalize_user_key_id():
    assert NovaAIGatewayLogService.normalize_user_key_id("alice") == "alice"
    expected = "hashed-" + hashlib.sha256(b"hashed-user").hexdigest()[:40]
    assert NovaAIGatewayLogService.normalize_user_key_id(" hashed-user ") == expected


def test_nova_aigateway_sample_rate_bounds():
    assert NovaAIGatewayLogService._normalize_sample_rate(None) == 1.0
    assert NovaAIGatewayLogService._normalize_sample_rate(0) == 0.0
    assert NovaAIGatewayLogService._normalize_sample_rate("0.25") == 0.25

    with pytest.raises(ValueError):
        NovaAIGatewayLogService._normalize_sample_rate(1.1)


def test_nova_aigateway_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        NovaAIGatewayLogService(base_url="http://nova.example", batch_size=0)


def test_nova_aigateway_enabled_config_supports_string_and_env_values(monkeypatch):
    monkeypatch.setenv("NOVA_AIGATEWAY_ENABLED", "true")

    assert NovaAIGatewayLogService.is_enabled_in_config({"enabled": True}) is True
    assert NovaAIGatewayLogService.is_enabled_in_config({"enabled": "true"}) is True
    assert (
        NovaAIGatewayLogService.is_enabled_in_config(
            {"enabled": "os.environ/NOVA_AIGATEWAY_ENABLED"}
        )
        is True
    )
    assert NovaAIGatewayLogService.is_enabled_in_config({"enabled": "false"}) is False


def test_nova_aigateway_rejects_non_positive_or_fractional_intervals():
    with pytest.raises(
        ValueError, match="flush_interval_seconds must be a positive integer"
    ):
        NovaAIGatewayLogService(
            base_url="http://nova.example", flush_interval_seconds=0
        )

    with pytest.raises(
        ValueError,
        match="public_keys_refresh_interval_seconds must be a positive integer",
    ):
        NovaAIGatewayLogService(
            base_url="http://nova.example",
            public_keys_refresh_interval_seconds=0.5,
        )


@pytest.mark.asyncio
async def test_nova_aigateway_uses_shorter_bootstrap_public_key_retry_interval():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        public_keys_refresh_interval_seconds=300,
    )

    try:
        assert service._get_public_key_refresh_sleep_interval() == 10
        service._has_successful_public_key_refresh = True
        assert service._get_public_key_refresh_sleep_interval() == 300
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_sampling_is_stable():
    service = NovaAIGatewayLogService(base_url="http://nova.example", sample_rate=0.5)

    try:
        standard_logging_object = _standard_logging_object()
        first = service._should_log_event(standard_logging_object)
        second = service._should_log_event(standard_logging_object)
        assert first is second
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_encrypts_with_per_user_per_day_dek():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    service = NovaAIGatewayLogService(base_url="http://nova.example")

    try:
        key_id = NovaAIGatewayLogService.normalize_user_key_id("alice")
        service._public_key_cache[key_id] = _build_public_key_record(
            key_id=key_id,
            pem=public_pem,
        )

        standard_logging_object = _standard_logging_object()
        first_item = service._encrypt_standard_logging_object(standard_logging_object)
        second_item = service._encrypt_standard_logging_object(standard_logging_object)

        assert first_item is not None
        assert second_item is not None
        assert first_item["wrapped_dek"] == second_item["wrapped_dek"]
        assert first_item["context"] == second_item["context"]

        wrapped_plaintext = private_key.decrypt(
            base64.b64decode(first_item["wrapped_dek"]),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        dek, context = wrapped_plaintext.rsplit(b"|", 1)
        assert context == b'{"date": "2026-04-21", "user": "alice"}'

        encrypted_bytes = base64.b64decode(first_item["encrypted_data"])
        decrypted_bytes = AESGCM(dek).decrypt(
            encrypted_bytes[:12], encrypted_bytes[12:], None
        )
        decrypted_payload = json.loads(decrypted_bytes.decode("utf-8"))
        assert decrypted_payload["id"] == "req-1"
        assert decrypted_payload["metadata"]["user_api_key_user_id"] == "alice"
        assert first_item["kms_key"].endswith("/cryptoKeyVersions/1")
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_rotates_dek_when_kms_version_changes():
    first_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    first_public_pem = (
        first_private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    second_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    second_public_pem = (
        second_private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    service = NovaAIGatewayLogService(base_url="http://nova.example")

    try:
        key_id = NovaAIGatewayLogService.normalize_user_key_id("alice")
        service._public_key_cache[key_id] = _build_public_key_record(
            key_id=key_id,
            pem=first_public_pem,
            version="projects/test/.../alice/cryptoKeyVersions/1",
        )

        standard_logging_object = _standard_logging_object()
        first_item = service._encrypt_standard_logging_object(standard_logging_object)

        service._public_key_cache[key_id] = _build_public_key_record(
            key_id=key_id,
            pem=second_public_pem,
            version="projects/test/.../alice/cryptoKeyVersions/2",
        )

        second_item = service._encrypt_standard_logging_object(standard_logging_object)

        assert first_item is not None
        assert second_item is not None
        assert first_item["kms_key"].endswith("/cryptoKeyVersions/1")
        assert second_item["kms_key"].endswith("/cryptoKeyVersions/2")
        assert first_item["wrapped_dek"] != second_item["wrapped_dek"]
        assert first_item["context"] == second_item["context"]
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_prunes_deks_from_old_dates():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    service = NovaAIGatewayLogService(base_url="http://nova.example")

    try:
        key_id = NovaAIGatewayLogService.normalize_user_key_id("alice")
        service._public_key_cache[key_id] = _build_public_key_record(
            key_id=key_id,
            pem=public_pem,
        )

        old_day_payload = _standard_logging_object(
            start_time=datetime(
                2026, 4, 20, 16, 35, 12, tzinfo=timezone.utc
            ).timestamp()
        )
        new_day_payload = _standard_logging_object(
            start_time=datetime(
                2026, 4, 21, 16, 35, 12, tzinfo=timezone.utc
            ).timestamp()
        )

        old_item = service._encrypt_standard_logging_object(old_day_payload)
        assert old_item is not None
        assert len(service._dek_cache) == 1

        new_item = service._encrypt_standard_logging_object(new_day_payload)
        assert new_item is not None
        assert len(service._dek_cache) == 1
        assert list(service._dek_cache.keys())[0][1] == "2026-04-21"
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_drops_event_without_internal_user():
    service = NovaAIGatewayLogService(base_url="http://nova.example")
    service.start = AsyncMock()  # type: ignore[method-assign]

    try:
        await service.enqueue(
            {
                "id": "req-1",
                "startTime": datetime.now(timezone.utc).timestamp(),
                "metadata": {},
            }
        )

        assert service.dropped_missing_user_count == 1
        assert service.log_queue == []
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_drops_sampled_out_event():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        sample_rate=0.0,
    )
    service.start = AsyncMock()  # type: ignore[method-assign]

    try:
        await service.enqueue(_standard_logging_object())

        assert service.dropped_sampled_out_count == 1
        assert service.log_queue == []
        service.start.assert_not_awaited()
    finally:
        await service.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_refreshes_public_keys_from_all_pages():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        public_keys_page_size=1,
    )
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.async_httpx_client.get = AsyncMock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "snapshotVersion": "v1",
                    "refreshedAt": "2026-04-21T16:35:12Z",
                    "items": [
                        {
                            "keyID": "alice",
                            "pem": public_pem,
                            "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                            "version": "projects/test/.../alice/cryptoKeyVersions/1",
                        }
                    ],
                    "page": 1,
                    "pageSize": 1,
                    "total": 2,
                    "totalPages": 2,
                },
                request=httpx.Request(
                    "GET", "http://nova.example/api/v1/kms/public-keys"
                ),
            ),
            httpx.Response(
                200,
                json={
                    "snapshotVersion": "v1",
                    "refreshedAt": "2026-04-21T16:35:12Z",
                    "items": [
                        {
                            "keyID": "bob",
                            "pem": public_pem,
                            "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                            "version": "projects/test/.../bob/cryptoKeyVersions/1",
                        }
                    ],
                    "page": 2,
                    "pageSize": 1,
                    "total": 2,
                    "totalPages": 2,
                },
                request=httpx.Request(
                    "GET", "http://nova.example/api/v1/kms/public-keys"
                ),
            ),
        ]
    )

    try:
        await service._refresh_public_keys()

        assert set(service._public_key_cache.keys()) == {"alice", "bob"}
        assert service._snapshot_version == "v1"
        assert service._snapshot_refreshed_at == "2026-04-21T16:35:12Z"
        assert service.async_httpx_client.get.await_count == 2
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_refresh_skips_invalid_public_key_entries():
    service = NovaAIGatewayLogService(base_url="http://nova.example")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.async_httpx_client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "snapshotVersion": "v1",
                "refreshedAt": "2026-04-21T16:35:12Z",
                "items": [
                    {
                        "keyID": "alice",
                        "pem": public_pem,
                        "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                        "version": "projects/test/.../alice/cryptoKeyVersions/1",
                    },
                    {
                        "keyID": "broken",
                        "pem": "not-a-pem",
                        "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                        "version": "projects/test/.../broken/cryptoKeyVersions/1",
                    },
                    {
                        "keyID": "missing-version",
                        "pem": public_pem,
                        "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                        "version": "",
                    },
                ],
                "page": 1,
                "pageSize": 2000,
                "total": 3,
                "totalPages": 1,
            },
            request=httpx.Request("GET", "http://nova.example/api/v1/kms/public-keys"),
        )
    )

    try:
        await service._refresh_public_keys()

        assert set(service._public_key_cache.keys()) == {"alice"}
        assert service._snapshot_version == "v1"
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_refresh_skips_non_object_public_key_entries():
    service = NovaAIGatewayLogService(base_url="http://nova.example")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.async_httpx_client.get = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "snapshotVersion": "v1",
                "refreshedAt": "2026-04-21T16:35:12Z",
                "items": [
                    None,
                    "broken",
                    {
                        "keyID": "alice",
                        "pem": public_pem,
                        "algorithm": "RSA_DECRYPT_OAEP_2048_SHA256",
                        "version": "projects/test/.../alice/cryptoKeyVersions/1",
                    },
                ],
                "page": 1,
                "pageSize": 2000,
                "total": 3,
                "totalPages": 1,
            },
            request=httpx.Request("GET", "http://nova.example/api/v1/kms/public-keys"),
        )
    )

    try:
        await service._refresh_public_keys()

        assert set(service._public_key_cache.keys()) == {"alice"}
        assert service._snapshot_version == "v1"
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_start_continues_when_initial_key_refresh_fails():
    service = NovaAIGatewayLogService(base_url="http://nova.example")
    service._refresh_public_keys = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("boom")
    )

    try:
        await service.start()

        assert service._startup_complete is True
        assert service._flush_task is not None
        assert service._public_key_refresh_task is not None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_nova_aigateway_batch_flush_runs_in_background_worker():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        batch_size=1,
        flush_interval_seconds=60,
    )
    service.start = AsyncMock()  # type: ignore[method-assign]
    service._encrypt_standard_logging_object = Mock(  # type: ignore[method-assign]
        return_value={
            "wrapped_dek": "wrapped",
            "context": "context",
            "encrypted_data": "cipher",
            "kms_key": "kms-key",
            "user": "alice",
        }
    )

    batch_started = asyncio.Event()
    release_batch = asyncio.Event()

    async def _send_batch_best_effort(batch):
        batch_started.set()
        await release_batch.wait()
        return None

    service._send_batch_best_effort = AsyncMock(  # type: ignore[method-assign]
        side_effect=_send_batch_best_effort
    )
    service._start_background_tasks()

    try:
        await asyncio.wait_for(service.enqueue(_standard_logging_object()), timeout=0.1)

        await asyncio.wait_for(batch_started.wait(), timeout=0.1)
        assert service.log_queue == []
        assert len(service._inflight_flush_tasks) == 1

        release_batch.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert service.log_queue == []
        assert service._inflight_flush_tasks == set()
    finally:
        release_batch.set()
        await service.stop()


@pytest.mark.asyncio
async def test_nova_aigateway_flush_queue_drops_items_on_failure():
    service = NovaAIGatewayLogService(base_url="http://nova.example", batch_size=2)
    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.log_queue = [
        {
            "wrapped_dek": "wrapped",
            "context": "context",
            "encrypted_data": "cipher",
            "kms_key": "kms-key",
            "user": "alice",
        }
    ]
    service.async_httpx_client.post = AsyncMock(
        return_value=httpx.Response(
            500,
            text="server error",
            request=httpx.Request(
                "POST", "http://nova.example/api/v1/logs/litellm/batches"
            ),
        )
    )

    try:
        await service._flush_queue()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert service.log_queue == []
        assert service.dropped_send_failure_count == 1
        assert service._inflight_flush_tasks == set()
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_flush_queue_sends_batch_and_clears_items():
    service = NovaAIGatewayLogService(base_url="http://nova.example", batch_size=2)
    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.log_queue = [
        {
            "wrapped_dek": "wrapped",
            "context": "context",
            "encrypted_data": "cipher",
            "kms_key": "kms-key",
            "user": "alice",
        }
    ]
    service.async_httpx_client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"batch_id": "batch-1", "duplicate": False},
            request=httpx.Request(
                "POST", "http://nova.example/api/v1/logs/litellm/batches"
            ),
        )
    )

    try:
        await service._flush_queue()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert service.log_queue == []
        post_kwargs = service.async_httpx_client.post.await_args.kwargs
        assert post_kwargs["headers"]["Content-Type"] == "application/json"
        assert post_kwargs["json"]["items"][0]["user"] == "alice"
        assert "batch_id" in post_kwargs["json"]
        assert "batch_created_at" in post_kwargs["json"]
        assert service._inflight_flush_tasks == set()
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_nova_aigateway_respects_max_inflight_batches():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        batch_size=1,
        flush_interval_seconds=60,
        max_inflight_batches=1,
    )
    service.start = AsyncMock()  # type: ignore[method-assign]
    service._encrypt_standard_logging_object = Mock(  # type: ignore[method-assign]
        return_value={
            "wrapped_dek": "wrapped",
            "context": "context",
            "encrypted_data": "cipher",
            "kms_key": "kms-key",
            "user": "alice",
        }
    )

    first_batch_started = asyncio.Event()
    release_batches = asyncio.Event()

    async def _send_batch_best_effort(batch):
        first_batch_started.set()
        await release_batches.wait()

    service._send_batch_best_effort = AsyncMock(  # type: ignore[method-assign]
        side_effect=_send_batch_best_effort
    )
    service._start_background_tasks()

    try:
        await service.enqueue(_standard_logging_object())
        await asyncio.wait_for(first_batch_started.wait(), timeout=0.1)

        await service.enqueue(_standard_logging_object(user_id="alice"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(service._inflight_flush_tasks) == 1
        assert len(service.log_queue) == 1

        release_batches.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert service.log_queue == []
    finally:
        release_batches.set()
        await service.stop()


@pytest.mark.asyncio
async def test_nova_aigateway_shutdown_drains_entire_queue():
    service = NovaAIGatewayLogService(
        base_url="http://nova.example",
        batch_size=1,
        max_inflight_batches=1,
        flush_interval_seconds=60,
    )
    original_client = service.async_httpx_client
    service.async_httpx_client = AsyncMock()
    service.log_queue = [
        {
            "wrapped_dek": f"wrapped-{idx}",
            "context": "context",
            "encrypted_data": "cipher",
            "kms_key": "kms-key",
            "user": "alice",
        }
        for idx in range(3)
    ]
    service.async_httpx_client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"batch_id": "batch-1", "duplicate": False},
            request=httpx.Request(
                "POST", "http://nova.example/api/v1/logs/litellm/batches"
            ),
        )
    )

    try:
        await service.stop()

        assert service.log_queue == []
        assert service.async_httpx_client.post.await_count == 3
    finally:
        await original_client.aclose()


@pytest.mark.asyncio
async def test_async_enqueue_proxy_nova_aigateway_log_uses_proxy_service():
    original_proxy_service = getattr(litellm, "proxy_nova_aigateway_service", None)
    proxy_service = AsyncMock()
    litellm.proxy_nova_aigateway_service = proxy_service

    try:
        payload = _standard_logging_object()
        await litellm_logging._async_enqueue_proxy_nova_aigateway_log(payload)

        enqueued_payload = proxy_service.enqueue.await_args.args[0]
        assert enqueued_payload == payload
        assert enqueued_payload is not payload
    finally:
        litellm.proxy_nova_aigateway_service = original_proxy_service


@pytest.mark.asyncio
async def test_async_enqueue_proxy_nova_aigateway_log_skips_payload_without_internal_user():
    original_proxy_service = getattr(litellm, "proxy_nova_aigateway_service", None)
    proxy_service = AsyncMock()
    litellm.proxy_nova_aigateway_service = proxy_service

    try:
        payload = _standard_logging_object()
        payload["metadata"] = {}
        await litellm_logging._async_enqueue_proxy_nova_aigateway_log(payload)

        proxy_service.enqueue.assert_not_awaited()
    finally:
        litellm.proxy_nova_aigateway_service = original_proxy_service


@pytest.mark.asyncio
async def test_async_enqueue_proxy_nova_aigateway_log_ignores_turn_off_message_logging():
    original_proxy_service = getattr(litellm, "proxy_nova_aigateway_service", None)
    original_turn_off_message_logging = litellm.turn_off_message_logging
    proxy_service = AsyncMock()
    litellm.proxy_nova_aigateway_service = proxy_service
    litellm.turn_off_message_logging = True

    try:
        payload = _standard_logging_object()
        await litellm_logging._async_enqueue_proxy_nova_aigateway_log(payload)

        enqueued_payload = proxy_service.enqueue.await_args.args[0]
        assert enqueued_payload["messages"] == payload["messages"]
        assert enqueued_payload["response"] == payload["response"]
        assert payload["messages"][0]["content"] == "hello"
    finally:
        litellm.proxy_nova_aigateway_service = original_proxy_service
        litellm.turn_off_message_logging = original_turn_off_message_logging
