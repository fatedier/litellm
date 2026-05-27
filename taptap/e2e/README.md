# TapTap E2E

This directory contains post-deploy E2E validation for TapTap local patches
against a deployed test environment and local mock providers

Nova AI Gateway Seedance validation design lives in
[taptap/docs/seedance-e2e.md](/Users/fate/local/projects/litellm/taptap/docs/seedance-e2e.md).

Principles:

- test logic should be deterministic and spec-driven
- AI agents can execute cases and summarize failures
- pass or fail should be decided by scripted assertions, not free-form agent judgment

Layout:

- `cases/`: structured test-case definitions
- `fixtures/`: environment-specific seed data or templates
- `scripts/`: execution wrappers used by CI or human operators

## Test suites

The E2E tests are split into two suites. The runner accepts one cases directory
at a time, so the two suites must be run separately

| Suite | Directory | Current coverage | Main requirements |
| --- | --- | --- | --- |
| Deployed environment | `cases/` | 17 cases, 21 scenarios | A running LiteLLM deployment, `TAPTAP_E2E_BASE_URL`, and `TAPTAP_E2E_MASTER_KEY` |
| Local mock provider | `local_cases/` | 4 cases, 8 scenarios | The deployed environment, two local mock providers, and the mock-provider environment variables below |

The default cases directory is `cases/`. Running `go run ./cmd/runner` does
not include `local_cases/`

## Running the deployed-environment suite

```bash
cd taptap/e2e
set -a; source .env.local; set +a
go run ./cmd/runner --list
go run ./cmd/runner
```

Up to four independent cases run concurrently by default while scenario and step order remain serial within each case:

```bash
go run ./cmd/runner
go run ./cmd/runner --jobs 1
```

`--jobs` defaults to `4`; set it to `1` for serial execution. Keep it bounded for the deployed environment because cases can share provider rate limits and gateway/database capacity. With `--fail-fast`, already running cases finish their cleanup, but no new cases are dispatched after the first failed case.

The runner summary counts top-level cases, not scenarios or individual HTTP
steps. Use `--list` to see every discovered case and scenario

## Running the local mock-provider suite

Start the two mock providers in separate terminals:

```bash
go run ./devtools/mock-llm-provider --name fail --addr :18081 --status 503
go run ./devtools/mock-llm-provider --name ok --addr :18082
```

In the runner terminal, load the shared environment and configure both provider
endpoints:

```bash
cd taptap/e2e
set -a; source .env.local; set +a

export TAPTAP_E2E_MOCK_PROVIDER_A_URL=http://localhost:18081
export TAPTAP_E2E_MOCK_PROVIDER_B_URL=http://localhost:18082
export TAPTAP_E2E_MOCK_PROVIDER_A_API_BASE=http://host.docker.internal:18081/v1
export TAPTAP_E2E_MOCK_PROVIDER_B_API_BASE=http://host.docker.internal:18082/v1
export TAPTAP_E2E_ENABLE_RETRY_DEPLOYMENT_FAILOVER=1
go run ./cmd/runner --cases-dir local_cases --list
go run ./cmd/runner --cases-dir local_cases --jobs 1
```

`*_URL` is used by the runner. `*_API_BASE` is written into LiteLLM deployments,
so it must be reachable from the proxy process.

Run the local suite serially because its cases reset and inspect the same two
mock-provider event stores. Parallel execution can mix events across cases and
produce false failures.

To run both suites, run the deployed-environment command first, then run the
local mock-provider command with `--cases-dir local_cases`

Environment:

- required: `TAPTAP_E2E_BASE_URL`
- required: `TAPTAP_E2E_MASTER_KEY`
- optional: `TAPTAP_E2E_SEEDANCE_PATH`

The runner creates and deletes temporary users, teams, and keys through the
master key. Case-specific credentials are not pre-provisioned.

Case schema:

- top level: `id`, `patch_id`, `variables`, `shared`, `scenarios`
- scenario: `id`, `requires_env`, `steps`, optional `cleanup`
- step: `request`, `media_probe`, `image_probe`, `frame_extract`, `cost_estimate`, `nova_discount_discovery`, or `cost_discount_assertion`, plus `expect` and optional `save`
- save item: `from` or `from_any`, plus `as`, optional `decode_base64_to`
- step can also use `frame_extract` for fixed key-frame generation

Supported request keys:

- `method`
- `path` or `url`
- `headers`
- `params`
- `json`
- `data`
- `body`
- `files`
- `timeout_seconds`
- `poll`
- `save_body_to`

Supported polling keys:

- `timeout_seconds`
- `interval_seconds`
- `success`
- `failure`
- `length_gte`
- `json_number_gte`
- `json_number_lte`

Supported `media_probe` keys:

- `path`

Supported `image_probe` keys:

- `path`

Supported `expect.image` keys:

- `format` (currently `png`)
- `width_gte`
- `height_gte`
- `transparent_pixels_gte`
- `transparent_fraction_gte` (0 to 1)

Supported `frame_extract` keys:

- `path`
- `outputs`

Supported `cost_estimate` keys:

- `usage`
- `pricing`
- `tolerance`

Supported `cost_estimate.usage` fields include:

- `input_text_tokens`
- `input_seconds`
- `input_characters`
- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `input_image_tokens`
- `output_text_tokens`
- `output_image_tokens`

Supported `cost_estimate.pricing` fields include:

- `input_cost_per_token`
- `input_cost_per_second`
- `input_cost_per_character`
- `cache_creation_input_token_cost`
- `cache_read_input_token_cost`
- `input_cost_per_image_token`
- `output_cost_per_token`
- `output_cost_per_image_token`

Supported `nova_discount_discovery` keys:

- `model_name`
- `master_key`
- `timeout_seconds`

Supported `cost_discount_assertion` keys:

- `expected_discount_percent`
- `discount_percent`
- `discount_amount`
- `original_cost`
- `total_cost`
- `tolerance`

Supported `repeat` helper keys:

- `text`
- `count`
- `separator`

Supported expectations:

- `status`
- `status_in`
- `json_equals`
- `json_not_equals`
- `json_number_gte`
- `json_number_lte`
- `json_contains`
- `json_exists`
- `header_equals`
- `header_exists`
- `body_contains`
- `media`

Interpolation:

- `${ENV_NAME}` from environment
- `${ENV_NAME:default}` with fallback
- `${saved_value}` from a prior `save` step within the same scenario
- `${run_id}` for unique temporary resource names per scenario run
- `${artifacts_dir}` for per-scenario temporary files
- `repeat:` helper map for generating long deterministic strings
