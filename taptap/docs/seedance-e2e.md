# Seedance Via Nova AI Gateway E2E

This document defines the intended end-to-end validation flow for Seedance
traffic routed through LiteLLM's `nova_aigateway` passthrough billing path.

## Goal

Validate the full deployed behavior of Seedance through Nova AI Gateway:

- the passthrough route accepts and forwards a task creation request
- the task can be polled until a terminal state
- spend logs contain the expected billing metadata
- the generated video artifact is structurally valid
- the generated video semantically matches a fixed prompt rubric

## Runtime Behavior To Preserve

The Nova AI Gateway passthrough path establishes these runtime expectations:

- Seedance traffic is configured as `passthrough_type: nova_aigateway`
- LiteLLM consumes Nova AI Gateway billing headers
- POST task creation requests write non-zero spend
- spend logs write `task_id`, `duration`, `ratio`,
  `output_width`, `output_height`, `output_fps`,
  `input_video_duration_seconds`, `billable_video_duration_seconds`,
  `output_tokens`, and `token_price_per_musd`
- GET task lookup routes are expected to be supported via the passthrough
  subpath route without writing task-creation spend

Relevant references:

- [taptap/e2e/cases/seedance_passthrough.yaml](/Users/fate/local/projects/litellm/taptap/e2e/cases/seedance_passthrough.yaml)
- billing handler test:
  `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_nova_aigateway_passthrough_logging_handler.py`
- passthrough route test:
  `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`

## Happy-Path Scenario

The recommended E2E scenario is a single full-flow case:

1. Create a temporary LiteLLM key with the proxy master key.
2. Submit a Seedance generation request through the passthrough route.
3. Assert the create response returns a task identifier.
4. Poll the task lookup route until the task reaches a terminal success state.
5. Query `/spend/logs` for the temporary key and assert billing metadata.
6. Extract the final video URL from the terminal task payload.
7. Download the video artifact.
8. Run structural media checks.
9. Run semantic checks against a fixed rubric.
10. Delete the temporary LiteLLM key.

This should remain one scenario so the evidence for routing, billing, and video
quality all comes from the same generated artifact.

## Request Pattern

### Create Task

- method: `POST`
- path: `/volcengine/api/v3/contents/generations/tasks`
- auth: LiteLLM temporary key
- expected response: `200`
- required save value: `task_id`

The E2E accepts `id` or `task_id` as valid task identifiers because provider
payloads can vary across environments.

### Poll Task

- method: `GET`
- path: `/volcengine/api/v3/contents/generations/tasks/{task_id}`
- auth: LiteLLM temporary key
- polling: repeat until a terminal success state or timeout

The exact response field names may vary slightly by upstream provider payload,
so the runner should support configurable success and failure JSON paths.

Recommended configurable paths:

- task state path candidates:
  - `status`
  - `data.status`
  - `task.status`
- terminal success values:
  - `succeeded`
  - `success`
  - `completed`
  - `done`
- terminal failure values:
  - `failed`
  - `error`

### Spend Log Lookup

- method: `GET`
- path: `/spend/logs`
- auth: proxy master key
- query: `api_key=<temporary_key>`

Required assertions:

- response contains the generated `task_id`
- response contains non-zero spend
- response contains `ratio`
- response contains `duration`

Preferred stronger assertions, if stable in the environment:

- `cost_tracking_strategy == seedance_video_generation`

## Video Artifact Validation

Video validation should have two layers.

### Structural Checks

These should be deterministic and non-AI:

- file downloads successfully
- container and stream are readable by `ffprobe`
- duration is present and close to the requested duration
- dimensions exist and match the billed ratio expectation closely
- there is at least one video stream

Recommended tools:

- `ffprobe` for metadata
- `ffmpeg` for frame extraction

### Semantic Checks

These should use a fixed rubric, not free-form human-style judgment.

Recommended flow:

1. Extract representative frames:
   - near the beginning
   - middle
   - near the end
2. Send the prompt, rubric, and frames to a multimodal judge.
3. Require structured JSON output.

The judge can be:

- a future LiteLLM multimodal model call
- an internal HTTP judge service
- a separate deterministic evaluator wrapper

The runner should treat the judge as an external classifier, not as a source of
open-ended prose.

## Prompt Design

Use a low-ambiguity prompt with easy-to-check motion and objects.

Recommended baseline prompt:

`A single bright red ball moves from left to right across a plain white background.`

Recommended rubric fields:

- `has_single_red_ball`
- `background_is_plain_white`
- `motion_is_left_to_right`
- `no_extra_primary_subjects`
- `semantic_pass`

Expected structured judge output:

```json
{
  "semantic_pass": true,
  "checks": {
    "has_single_red_ball": true,
    "background_is_plain_white": true,
    "motion_is_left_to_right": true,
    "no_extra_primary_subjects": true
  }
}
```

## Runner Capabilities Used

The executable case relies on these Go runner capabilities:

- polling support for HTTP steps
  - retry interval
  - max wait time
  - success and failure JSON path matching
- binary download support
  - save response body to a file path
- external media inspection support
  - run `ffprobe`
  - optionally run `ffmpeg` frame extraction
- semantic judge integration hook
  - structured request/response contract
  - pass/fail based on JSON fields, not prose

## Practical Rollout Order

Keep validation staged when extending this case.

### Stage 1

- submit task
- poll task success
- verify spend logs

This is enough to validate the passthrough route and Nova AI Gateway billing path.

### Stage 2

- download artifact
- run `ffprobe`
- check duration and dimensions

This validates the actual media artifact.

### Stage 3

- extract frames
- run semantic judge

This adds content-level confidence, but should come after Stage 1 and Stage 2
are already stable.

## Current Recommendation

Maintain the Seedance E2E in this order:

1. Keep task submission, task polling, and spend log assertions stable.
2. Keep `ffprobe`-based structural checks deterministic.
3. Add semantic judging last, behind a clean interface.

This keeps the executable case useful without forcing a semantic judge
dependency into the core billing and routing check.
