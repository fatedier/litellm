# Scripts

Place E2E helper scripts here.

Expected future entry points:

- a seeded environment setup helper
- a deterministic local-patch E2E runner
- CI wrappers that publish test evidence and summaries

Current execution entry point lives in the Go runner:

- `../cmd/runner`

Example:

```bash
cd taptap/e2e
go run ./cmd/runner --list
go run ./cmd/runner --patch-id utc_daily_user_budget_guardrail
```
