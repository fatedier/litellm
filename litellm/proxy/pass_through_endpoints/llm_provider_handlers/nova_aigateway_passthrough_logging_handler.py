import base64
import binascii
import json
import math
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    NOVA_AIGATEWAY_BILLING_HEADER_NAME,
)
from litellm.types.utils import StandardPassThroughResponseObject


class NovaAIGatewayPassthroughLoggingHandler:
    SKIP_PASSTHROUGH_SUCCESS_LOGGING_KWARG = (
        "_skip_nova_aigateway_passthrough_success_logging"
    )

    @staticmethod
    def nova_aigateway_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        del url_route, result, cache_hit

        if httpx_response.status_code < 200 or httpx_response.status_code >= 300:
            return {"result": None, "kwargs": kwargs}

        billing_payload = NovaAIGatewayPassthroughLoggingHandler._get_billing_payload(
            httpx_response=httpx_response
        )
        if billing_payload is None:
            NovaAIGatewayPassthroughLoggingHandler._set_skip_logging(kwargs=kwargs)
            return {"result": None, "kwargs": kwargs}

        cost = NovaAIGatewayPassthroughLoggingHandler._parse_cost(
            billing_payload.get("cost")
        )
        if cost is None:
            NovaAIGatewayPassthroughLoggingHandler._set_skip_logging(kwargs=kwargs)
            verbose_proxy_logger.warning(
                "nova_aigateway passthrough billing header missing valid cost"
            )
            return {"result": None, "kwargs": kwargs}

        model = NovaAIGatewayPassthroughLoggingHandler._parse_string_field(
            billing_payload.get("model")
        )
        if model is None and isinstance(request_body, dict):
            model = NovaAIGatewayPassthroughLoggingHandler._parse_string_field(
                request_body.get("model")
            )

        task_id = NovaAIGatewayPassthroughLoggingHandler._parse_string_field(
            billing_payload.get("task_id")
        )
        custom_llm_provider = (
            NovaAIGatewayPassthroughLoggingHandler._parse_string_field(
                billing_payload.get("custom_llm_provider")
            )
        )

        spend_logs_metadata: Dict[str, Any] = {}
        billing_metadata = billing_payload.get("metadata")
        if isinstance(billing_metadata, dict):
            spend_logs_metadata.update(billing_metadata)
        if task_id is not None:
            spend_logs_metadata["task_id"] = task_id
        if spend_logs_metadata:
            NovaAIGatewayPassthroughLoggingHandler._merge_spend_logs_metadata(
                kwargs=kwargs,
                spend_logs_metadata=spend_logs_metadata,
            )

        if model is not None:
            kwargs["model"] = model
            NovaAIGatewayPassthroughLoggingHandler._set_model_group_metadata(
                kwargs=kwargs,
                model=model,
            )
            logging_obj.model_call_details["model"] = model

        if custom_llm_provider is not None:
            kwargs["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider

        kwargs["response_cost"] = cost
        logging_obj.model_call_details["response_cost"] = cost

        standard_logging_object = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=StandardPassThroughResponseObject(response=response_body),
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )
        kwargs["standard_logging_object"] = standard_logging_object
        logging_obj.model_call_details["standard_logging_object"] = (
            standard_logging_object
        )
        logging_obj.model_call_details["litellm_params"] = kwargs.get(
            "litellm_params", {}
        )

        return {"result": None, "kwargs": kwargs}

    @staticmethod
    def _get_billing_payload(
        httpx_response: httpx.Response,
    ) -> Optional[Dict[str, Any]]:
        header_value = httpx_response.headers.get(NOVA_AIGATEWAY_BILLING_HEADER_NAME)
        if not isinstance(header_value, str) or not header_value.strip():
            verbose_proxy_logger.debug(
                "nova_aigateway passthrough response missing billing header"
            )
            return None

        try:
            payload_bytes = NovaAIGatewayPassthroughLoggingHandler._decode_base64url(
                header_value.strip()
            )
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (
            binascii.Error,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as e:
            verbose_proxy_logger.warning(
                "Failed to parse nova_aigateway passthrough billing header: %s",
                str(e),
            )
            return None

        if not isinstance(payload, dict):
            verbose_proxy_logger.warning(
                "nova_aigateway passthrough billing header payload must be a JSON object"
            )
            return None

        return payload

    @staticmethod
    def _decode_base64url(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    @staticmethod
    def _parse_cost(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        cost = float(value)
        if not math.isfinite(cost) or cost < 0:
            return None
        return cost

    @staticmethod
    def _parse_string_field(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value
        return None

    @staticmethod
    def _merge_spend_logs_metadata(kwargs: dict, spend_logs_metadata: dict) -> None:
        litellm_params = kwargs.setdefault("litellm_params", {})
        metadata = litellm_params.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            litellm_params["metadata"] = metadata

        existing_spend_logs_metadata = metadata.get("spend_logs_metadata")
        if not isinstance(existing_spend_logs_metadata, dict):
            existing_spend_logs_metadata = {}

        existing_spend_logs_metadata.update(spend_logs_metadata)
        metadata["spend_logs_metadata"] = existing_spend_logs_metadata

    @staticmethod
    def _set_skip_logging(kwargs: dict) -> None:
        kwargs[
            NovaAIGatewayPassthroughLoggingHandler.SKIP_PASSTHROUGH_SUCCESS_LOGGING_KWARG
        ] = True

    @staticmethod
    def _set_model_group_metadata(kwargs: dict, model: str) -> None:
        litellm_params = kwargs.setdefault("litellm_params", {})
        metadata = litellm_params.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            litellm_params["metadata"] = metadata

        metadata["model_group"] = model
