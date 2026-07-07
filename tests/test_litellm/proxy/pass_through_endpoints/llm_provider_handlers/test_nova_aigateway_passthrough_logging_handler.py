import base64
import json
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock

import httpx

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.nova_aigateway_passthrough_logging_handler import (
    NovaAIGatewayPassthroughLoggingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    NOVA_AIGATEWAY_BILLING_HEADER_NAME,
    PassthroughStandardLoggingPayload,
)


def _encode_billing_payload(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode(
        "utf-8"
    )
    return encoded.rstrip("=")


class TestNovaAIGatewayPassthroughLoggingHandler:
    def setup_method(self):
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.request_body = {"prompt": "create a video"}
        self.response_body = {"output": {"task_id": "task-123"}}

    def _create_response(self, billing_payload: Optional[dict]) -> httpx.Response:
        headers = {}
        if billing_payload is not None:
            headers[NOVA_AIGATEWAY_BILLING_HEADER_NAME] = _encode_billing_payload(
                billing_payload
            )
        return httpx.Response(
            200,
            headers=headers,
            json=self.response_body,
            request=httpx.Request("POST", "https://nova-aigateway.example.com/tasks"),
        )

    def _create_logging_obj(self):
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.cost_breakdown = None
        return mock_logging_obj

    def _create_kwargs(self):
        return {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url="https://nova-aigateway.example.com/tasks",
                request_body=self.request_body,
                request_method="POST",
                passthrough_type="nova_aigateway",
            ),
            "litellm_params": {"metadata": {}},
            "call_type": "pass_through_endpoint",
        }

    def test_sets_cost_model_and_task_metadata(self):
        mock_logging_obj = self._create_logging_obj()
        kwargs = self._create_kwargs()
        response = self._create_response(
            {
                "cost": 1.25,
                "model": "doubao-seedance-2-0-260128",
                "task_id": "task-123",
            }
        )

        result = (
            NovaAIGatewayPassthroughLoggingHandler.nova_aigateway_passthrough_handler(
                httpx_response=response,
                response_body=self.response_body,
                logging_obj=mock_logging_obj,
                url_route="https://nova-aigateway.example.com/tasks",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body=self.request_body,
                **kwargs,
            )
        )

        result_kwargs = result["kwargs"]
        metadata = result_kwargs["litellm_params"]["metadata"]
        assert result["result"] is None
        assert result_kwargs["response_cost"] == 1.25
        assert result_kwargs["model"] == "doubao-seedance-2-0-260128"
        assert metadata["model_group"] == "doubao-seedance-2-0-260128"
        assert metadata["spend_logs_metadata"] == {"task_id": "task-123"}
        assert mock_logging_obj.model_call_details["response_cost"] == 1.25
        assert (
            mock_logging_obj.model_call_details["model"] == "doubao-seedance-2-0-260128"
        )
        assert "standard_logging_object" in result_kwargs

    def test_merges_billing_metadata_and_custom_provider(self):
        mock_logging_obj = self._create_logging_obj()
        kwargs = self._create_kwargs()
        kwargs["litellm_params"]["metadata"]["spend_logs_metadata"] = {
            "existing": "kept",
            "task_id": "existing-task",
        }
        response = self._create_response(
            {
                "cost": 2.5,
                "model": "doubao-seedance-2-0-260128",
                "task_id": "top-level-task",
                "custom_llm_provider": "volcengine-ark",
                "metadata": {
                    "duration": 5,
                    "resolution": "1080p",
                    "task_id": "metadata-task",
                },
            }
        )

        result = (
            NovaAIGatewayPassthroughLoggingHandler.nova_aigateway_passthrough_handler(
                httpx_response=response,
                response_body=self.response_body,
                logging_obj=mock_logging_obj,
                url_route="https://nova-aigateway.example.com/tasks",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body=self.request_body,
                **kwargs,
            )
        )

        result_kwargs = result["kwargs"]
        spend_logs_metadata = result_kwargs["litellm_params"]["metadata"][
            "spend_logs_metadata"
        ]
        assert result_kwargs["custom_llm_provider"] == "volcengine-ark"
        assert (
            mock_logging_obj.model_call_details["custom_llm_provider"]
            == "volcengine-ark"
        )
        assert spend_logs_metadata == {
            "existing": "kept",
            "duration": 5,
            "resolution": "1080p",
            "task_id": "top-level-task",
        }

    def test_missing_header_marks_logging_skipped(self):
        mock_logging_obj = self._create_logging_obj()
        kwargs = self._create_kwargs()
        response = self._create_response(billing_payload=None)

        result = (
            NovaAIGatewayPassthroughLoggingHandler.nova_aigateway_passthrough_handler(
                httpx_response=response,
                response_body=self.response_body,
                logging_obj=mock_logging_obj,
                url_route="https://nova-aigateway.example.com/tasks",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body=self.request_body,
                **kwargs,
            )
        )

        assert "response_cost" not in result["kwargs"]
        assert (
            result["kwargs"][
                NovaAIGatewayPassthroughLoggingHandler.SKIP_PASSTHROUGH_SUCCESS_LOGGING_KWARG
            ]
            is True
        )
        assert result["kwargs"]["litellm_params"]["metadata"] == {}
        assert "response_cost" not in mock_logging_obj.model_call_details

    def test_invalid_cost_marks_logging_skipped(self):
        mock_logging_obj = self._create_logging_obj()
        kwargs = self._create_kwargs()
        response = self._create_response(
            {
                "cost": -1.0,
                "model": "doubao-seedance-2-0-260128",
                "task_id": "task-123",
            }
        )

        result = (
            NovaAIGatewayPassthroughLoggingHandler.nova_aigateway_passthrough_handler(
                httpx_response=response,
                response_body=self.response_body,
                logging_obj=mock_logging_obj,
                url_route="https://nova-aigateway.example.com/tasks",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body=self.request_body,
                **kwargs,
            )
        )

        result_kwargs = result["kwargs"]
        assert "response_cost" not in result_kwargs
        assert "response_cost" not in mock_logging_obj.model_call_details
        assert "model" not in result_kwargs
        assert (
            result_kwargs[
                NovaAIGatewayPassthroughLoggingHandler.SKIP_PASSTHROUGH_SUCCESS_LOGGING_KWARG
            ]
            is True
        )
        assert result_kwargs["litellm_params"]["metadata"] == {}
