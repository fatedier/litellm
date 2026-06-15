import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import StandardLoggingPayload

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError as exc:
    hashes = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    padding = None  # type: ignore[assignment]
    _CRYPTOGRAPHY_IMPORT_ERROR = exc
else:
    _CRYPTOGRAPHY_IMPORT_ERROR = None


_CRYPTO_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,63}$")
_HASHED_CRYPTO_KEY_ID_PREFIX = "hashed-"
_CRYPTO_KEY_ID_HASH_LENGTH = 40


@dataclass(frozen=True)
class _PublicKeyRecord:
    key_id: str
    pem: str
    version: str
    algorithm: str
    public_key: Any


@dataclass(frozen=True)
class _DEKRecord:
    dek: bytes
    wrapped_dek_b64: str
    context_b64: str
    kms_key: str


class NovaAIGatewayLogService:
    DEFAULT_PUBLIC_KEYS_PATH = "/api/v1/kms/public-keys"
    DEFAULT_LOG_BATCHES_PATH = "/api/v1/logs/litellm/batches"
    DEFAULT_BOOTSTRAP_PUBLIC_KEYS_RETRY_INTERVAL_SECONDS = 10

    def __init__(
        self,
        *,
        base_url: str,
        access_token: Optional[str] = None,
        public_keys_path: str = DEFAULT_PUBLIC_KEYS_PATH,
        log_batches_path: str = DEFAULT_LOG_BATCHES_PATH,
        public_keys_page_size: int = 2000,
        public_keys_refresh_interval_seconds: int = 300,
        batch_size: int = 100,
        flush_interval_seconds: int = 5,
        request_timeout_seconds: int = 10,
        max_pending_items: int = 10000,
        max_inflight_batches: int = 4,
        sample_rate: float = 1.0,
    ) -> None:
        self._raise_if_crypto_unavailable()

        self.base_url = str(base_url or "").rstrip("/")
        self.access_token = access_token
        self.public_keys_path = str(public_keys_path or self.DEFAULT_PUBLIC_KEYS_PATH)
        self.log_batches_path = str(log_batches_path or self.DEFAULT_LOG_BATCHES_PATH)
        self.public_keys_page_size = int(public_keys_page_size)
        self.public_keys_refresh_interval_seconds = self._normalize_positive_interval(
            public_keys_refresh_interval_seconds,
            "public_keys_refresh_interval_seconds",
        )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.flush_interval = self._normalize_positive_interval(flush_interval_seconds, "flush_interval_seconds")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_pending_items = int(max_pending_items)
        self.max_inflight_batches = int(max_inflight_batches)
        if self.max_inflight_batches <= 0:
            raise ValueError("max_inflight_batches must be greater than 0")
        self.sample_rate = self._normalize_sample_rate(sample_rate)

        self._queue_lock = asyncio.Lock()
        self._startup_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._flush_requested = asyncio.Event()
        self._flush_task: Optional[asyncio.Task[Any]] = None
        self._public_key_refresh_task: Optional[asyncio.Task[Any]] = None
        self._startup_complete = False
        self._shutdown_started = False
        self._has_successful_public_key_refresh = False
        self._snapshot_version: Optional[str] = None
        self._snapshot_refreshed_at: Optional[str] = None
        self._public_key_cache: Dict[str, _PublicKeyRecord] = {}
        self._dek_cache: Dict[Tuple[str, str, str], _DEKRecord] = {}
        self._inflight_flush_tasks: Set[asyncio.Task[Any]] = set()
        self.last_flush_time = time.time()

        self.dropped_missing_user_count = 0
        self.dropped_missing_public_key_count = 0
        self.dropped_queue_full_count = 0
        self.dropped_sampled_out_count = 0
        self.dropped_send_failure_count = 0

        self.async_httpx_client = httpx.AsyncClient(timeout=self.request_timeout_seconds)
        self.log_queue: List[Dict[str, str]] = []

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "NovaAIGatewayLogService":
        # This service is configured from process-local config/env only. It is not
        # intended to participate in DB/UI-backed litellm_settings sync or hot reload.
        resolved_config = cls._resolve_config(config)
        resolved_config.pop("enabled", None)
        base_url = str(resolved_config.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("litellm_settings.nova_aigateway requires `base_url` to be configured")
        return cls(**resolved_config)

    async def start(self) -> None:
        if self._startup_complete:
            return

        async with self._startup_lock:
            if self._startup_complete:
                return
            # Best-effort bootstrap: startup waits for one bounded public-key fetch
            # attempt so the service can come up ready when Nova is healthy, but a
            # timeout/failure only degrades audit logging instead of blocking startup.
            try:
                await self._refresh_public_keys()
            except Exception as exc:
                verbose_logger.exception(
                    "Nova AI Gateway service failed initial public key refresh; continuing in degraded mode: %s",
                    str(exc),
                )
            self._start_background_tasks()
            self._startup_complete = True

    async def stop(self) -> None:
        async with self._startup_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        tasks = [self._flush_task, self._public_key_refresh_task]
        for task in tasks:
            if task is not None:
                task.cancel()
        if tasks:
            await asyncio.gather(
                *[task for task in tasks if task is not None],
                return_exceptions=True,
            )

        try:
            while True:
                await self._flush_queue()
                if self._inflight_flush_tasks:
                    await asyncio.gather(
                        *list(self._inflight_flush_tasks),
                        return_exceptions=True,
                    )

                async with self._queue_lock:
                    if not self.log_queue and not self._inflight_flush_tasks:
                        break
        finally:
            await self.async_httpx_client.aclose()

    async def enqueue(self, standard_logging_object: StandardLoggingPayload) -> None:
        try:
            if not self._should_log_event(standard_logging_object):
                self.dropped_sampled_out_count += 1
                return

            await self.start()

            encrypted_item = self._encrypt_standard_logging_object(standard_logging_object)
            if encrypted_item is None:
                return

            should_flush = False
            async with self._queue_lock:
                if len(self.log_queue) >= self.max_pending_items:
                    self.dropped_queue_full_count += 1
                    verbose_logger.error(
                        "Nova AI Gateway service dropping log because queue is full (max_pending_items=%s)",
                        self.max_pending_items,
                    )
                    return
                self.log_queue.append(encrypted_item)
                should_flush = len(self.log_queue) >= self.batch_size

            if should_flush:
                self._request_flush()
        except Exception as exc:
            verbose_logger.exception("Nova AI Gateway service failed to process log event: %s", str(exc))

    async def _flush_queue(self) -> None:
        async with self._flush_lock:
            while True:
                if len(self._inflight_flush_tasks) >= self.max_inflight_batches:
                    return

                async with self._queue_lock:
                    if not self.log_queue:
                        return
                    batch = list(self.log_queue[: self.batch_size])
                    del self.log_queue[: len(batch)]

                self._schedule_batch_send(batch=batch)
                self.last_flush_time = time.time()

    def _schedule_batch_send(self, batch: List[Dict[str, str]]) -> None:
        task = asyncio.create_task(
            self._send_batch_best_effort(batch=batch),
            name="litellm-nova-aigateway-send-batch",
        )
        self._inflight_flush_tasks.add(task)
        task.add_done_callback(self._on_batch_send_done)

    def _on_batch_send_done(self, task: asyncio.Task[Any]) -> None:
        self._inflight_flush_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            verbose_logger.exception(
                "Nova AI Gateway service send task failed unexpectedly: %s",
                str(exc),
            )

        if not self._shutdown_started:
            self._request_flush()

    async def _send_batch_best_effort(self, batch: List[Dict[str, str]]) -> None:
        if not batch:
            return

        payload = {
            "batch_id": str(uuid.uuid4()),
            "batch_created_at": self._utc_now_rfc3339(),
            "items": batch,
        }

        try:
            response = await self.async_httpx_client.post(
                self._build_url(self.log_batches_path),
                headers=self._build_headers(include_json_content_type=True),
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError:
            self.dropped_send_failure_count += len(batch)
            verbose_logger.error(
                "Nova AI Gateway service failed to upload batch: status=%s body=%s",
                response.status_code,
                response.text,
            )
        except Exception as exc:
            self.dropped_send_failure_count += len(batch)
            verbose_logger.exception("Nova AI Gateway service failed to upload batch: %s", str(exc))

    def _start_background_tasks(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(
                self._flush_worker_loop(),
                name="litellm-nova-aigateway-flush",
            )
        if self._public_key_refresh_task is None or self._public_key_refresh_task.done():
            self._public_key_refresh_task = asyncio.create_task(
                self._periodic_public_key_refresh(),
                name="litellm-nova-aigateway-public-keys-refresh",
            )

    def _request_flush(self) -> None:
        self._flush_requested.set()

    async def _flush_worker_loop(self) -> None:
        while True:
            triggered_by_event = False
            try:
                await asyncio.wait_for(
                    self._flush_requested.wait(),
                    timeout=self.flush_interval,
                )
                triggered_by_event = True
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

            if triggered_by_event:
                self._flush_requested.clear()

            try:
                await self._flush_queue()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                verbose_logger.exception(
                    "Nova AI Gateway service failed to flush queued logs: %s",
                    str(exc),
                )

    async def _periodic_public_key_refresh(self) -> None:
        while True:
            await asyncio.sleep(self._get_public_key_refresh_sleep_interval())
            try:
                await self._refresh_public_keys()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                verbose_logger.exception(
                    "Nova AI Gateway service failed to refresh public keys: %s",
                    str(exc),
                )

    async def _refresh_public_keys(self) -> None:
        async with self._refresh_lock:
            next_cache: Dict[str, _PublicKeyRecord] = {}
            page = 1
            total_pages = 1
            snapshot_version: Optional[str] = None
            refreshed_at: Optional[str] = None

            while page <= total_pages:
                response_payload = await self._fetch_public_keys_page(page=page)
                snapshot_version = cast(Optional[str], response_payload.get("snapshotVersion"))
                refreshed_at = cast(Optional[str], response_payload.get("refreshedAt"))
                total_pages = int(response_payload.get("totalPages") or 1)

                for item in response_payload.get("items", []) or []:
                    try:
                        key_id = str(item.get("keyID") or "").strip()
                        pem = str(item.get("pem") or "").strip()
                        version = str(item.get("version") or "").strip()
                        algorithm = str(item.get("algorithm") or "").strip()
                        if not key_id or not pem or not version:
                            raise ValueError("Nova AI Gateway public key snapshot contains an invalid item")

                        public_key = serialization.load_pem_public_key(pem.encode("utf-8"))
                        next_cache[key_id] = _PublicKeyRecord(
                            key_id=key_id,
                            pem=pem,
                            version=version,
                            algorithm=algorithm,
                            public_key=public_key,
                        )
                    except Exception as exc:
                        invalid_key_id = item.get("keyID") if isinstance(item, dict) else None
                        verbose_logger.warning(
                            "Nova AI Gateway service skipping invalid public key entry key_id=%s: %s",
                            invalid_key_id,
                            str(exc),
                        )

                page += 1

            self._public_key_cache = next_cache
            self._has_successful_public_key_refresh = True
            self._snapshot_version = snapshot_version
            self._snapshot_refreshed_at = refreshed_at
            verbose_logger.debug(
                "Nova AI Gateway service refreshed %s public keys (snapshot_version=%s)",
                len(next_cache),
                snapshot_version,
            )

    async def _fetch_public_keys_page(self, page: int) -> Dict[str, Any]:
        response = await self.async_httpx_client.get(
            self._build_url(self.public_keys_path),
            headers=self._build_headers(),
            params={"page": page, "pageSize": self.public_keys_page_size},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Nova AI Gateway public keys response must be a JSON object")
        return payload

    def _encrypt_standard_logging_object(
        self, standard_logging_object: StandardLoggingPayload
    ) -> Optional[Dict[str, str]]:
        user_id = self._get_internal_user_id(standard_logging_object)
        if user_id is None:
            self.dropped_missing_user_count += 1
            verbose_logger.error(
                "Nova AI Gateway service dropping log because metadata.user_api_key_user_id is missing"
            )
            return None

        normalized_key_id = self.normalize_user_key_id(user_id)
        public_key_record = self._public_key_cache.get(normalized_key_id)
        if public_key_record is None:
            self.dropped_missing_public_key_count += 1
            verbose_logger.error(
                "Nova AI Gateway service dropping log because public key is missing for user_id=%s normalized_key_id=%s",
                user_id,
                normalized_key_id,
            )
            return None

        log_date = self._get_log_date(standard_logging_object)
        dek_record = self._get_or_create_dek_record(
            normalized_key_id=normalized_key_id,
            user_id=user_id,
            log_date=log_date,
            public_key_record=public_key_record,
        )

        encrypted_data = self._encrypt_payload_bytes(
            dek=dek_record.dek,
            payload_bytes=safe_dumps(standard_logging_object).encode("utf-8"),
        )
        return {
            "wrapped_dek": dek_record.wrapped_dek_b64,
            "context": dek_record.context_b64,
            "encrypted_data": base64.b64encode(encrypted_data).decode("utf-8"),
            "kms_key": dek_record.kms_key,
            "user": user_id,
        }

    def _should_log_event(self, standard_logging_object: StandardLoggingPayload) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False

        sample_key = self._get_sample_key(standard_logging_object)
        digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
        sample_value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return sample_value < self.sample_rate

    @staticmethod
    def _get_sample_key(standard_logging_object: StandardLoggingPayload) -> str:
        for key in ("id", "trace_id"):
            value = standard_logging_object.get(key)
            if value is not None:
                normalized = str(value).strip()
                if normalized:
                    return normalized

        start_time = standard_logging_object.get("startTime")
        if start_time is not None:
            return str(start_time)
        return safe_dumps(standard_logging_object)

    def _get_or_create_dek_record(
        self,
        normalized_key_id: str,
        user_id: str,
        log_date: str,
        public_key_record: _PublicKeyRecord,
    ) -> _DEKRecord:
        self._prune_dek_cache(current_log_date=log_date)
        cache_key = (normalized_key_id, log_date, public_key_record.version)
        dek_record = self._dek_cache.get(cache_key)
        if dek_record is not None:
            return dek_record

        context = self._build_context(user_id=user_id, log_date=log_date)
        dek = secrets.token_bytes(32)
        wrapped_dek = public_key_record.public_key.encrypt(
            dek + b"|" + context,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        dek_record = _DEKRecord(
            dek=dek,
            wrapped_dek_b64=base64.b64encode(wrapped_dek).decode("utf-8"),
            context_b64=base64.b64encode(context).decode("utf-8"),
            kms_key=public_key_record.version,
        )
        self._dek_cache[cache_key] = dek_record
        return dek_record

    def _prune_dek_cache(self, current_log_date: str) -> None:
        stale_cache_keys = [cache_key for cache_key in self._dek_cache.keys() if cache_key[1] != current_log_date]
        for cache_key in stale_cache_keys:
            self._dek_cache.pop(cache_key, None)

    def _get_public_key_refresh_sleep_interval(self) -> int:
        if self._has_successful_public_key_refresh:
            return self.public_keys_refresh_interval_seconds
        return min(
            self.DEFAULT_BOOTSTRAP_PUBLIC_KEYS_RETRY_INTERVAL_SECONDS,
            self.public_keys_refresh_interval_seconds,
        )

    def _encrypt_payload_bytes(self, dek: bytes, payload_bytes: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(dek)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, payload_bytes, None)
        return nonce + ciphertext

    def _build_headers(self, include_json_content_type: bool = False) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if include_json_content_type:
            headers["Content-Type"] = "application/json"
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not self.base_url:
            raise ValueError("litellm_settings.nova_aigateway requires `base_url` to be configured")
        if path.startswith("/"):
            return f"{self.base_url}{path}"
        return f"{self.base_url}/{path}"

    @classmethod
    def normalize_user_key_id(cls, user_id: str) -> str:
        normalized_user_id = user_id.strip()
        if _CRYPTO_KEY_ID_PATTERN.match(normalized_user_id) and not normalized_user_id.startswith(
            _HASHED_CRYPTO_KEY_ID_PREFIX
        ):
            return normalized_user_id

        digest = hashlib.sha256(normalized_user_id.encode("utf-8")).hexdigest()
        return _HASHED_CRYPTO_KEY_ID_PREFIX + digest[:_CRYPTO_KEY_ID_HASH_LENGTH]

    @staticmethod
    def _get_internal_user_id(
        standard_logging_object: StandardLoggingPayload,
    ) -> Optional[str]:
        metadata = standard_logging_object.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None

        user_id = metadata.get("user_api_key_user_id")
        if user_id is None:
            return None

        normalized = str(user_id).strip()
        return normalized or None

    @staticmethod
    def _get_log_date(standard_logging_object: StandardLoggingPayload) -> str:
        start_time = standard_logging_object.get("startTime")
        if isinstance(start_time, (int, float)):
            return datetime.fromtimestamp(float(start_time), tz=timezone.utc).strftime("%Y-%m-%d")
        if isinstance(start_time, str):
            try:
                parsed = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _build_context(user_id: str, log_date: str) -> bytes:
        return json.dumps({"date": log_date, "user": user_id}, sort_keys=True).encode("utf-8")

    @staticmethod
    def _utc_now_rfc3339() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @classmethod
    def _resolve_config(cls, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        resolved_config = {
            "public_keys_path": cls.DEFAULT_PUBLIC_KEYS_PATH,
            "log_batches_path": cls.DEFAULT_LOG_BATCHES_PATH,
            "public_keys_page_size": 2000,
            "public_keys_refresh_interval_seconds": 300,
            "batch_size": 100,
            "flush_interval_seconds": 5,
            "request_timeout_seconds": 10,
            "max_pending_items": 10000,
            "max_inflight_batches": 4,
            "sample_rate": 1.0,
        }
        if config:
            resolved_config.update(config)

        for key, value in list(resolved_config.items()):
            resolved_config[key] = cls._resolve_secret_value(value)

        return resolved_config

    @classmethod
    def is_enabled_in_config(cls, config: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(config, dict):
            return False

        enabled = cls._resolve_secret_value(config.get("enabled"))
        if isinstance(enabled, bool):
            return enabled
        if isinstance(enabled, str):
            return str_to_bool(enabled) is True
        return False

    @staticmethod
    def _resolve_secret_value(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("os.environ/"):
            return litellm.get_secret(value)
        return value

    @staticmethod
    def _normalize_sample_rate(value: Any) -> float:
        sample_rate = float(value if value is not None else 1.0)
        if sample_rate < 0.0 or sample_rate > 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        return sample_rate

    @staticmethod
    def _normalize_positive_interval(value: Any, field_name: str) -> int:
        interval = float(value)
        if interval <= 0 or not interval.is_integer():
            raise ValueError(f"{field_name} must be a positive integer")
        return int(interval)

    @staticmethod
    def _raise_if_crypto_unavailable() -> None:
        if _CRYPTOGRAPHY_IMPORT_ERROR is not None:
            raise ImportError(
                "cryptography is required for litellm_settings.nova_aigateway"
            ) from _CRYPTOGRAPHY_IMPORT_ERROR
