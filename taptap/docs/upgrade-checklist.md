# Local Patch Upgrade Checklist

Use this checklist when rebasing or porting TapTap local patches onto a new
upstream LiteLLM release.

## 1. Inventory

- Confirm upstream target tag and local destination branch.
- Review the root `Dockerfile` and its pinned build dependencies.
- Review `taptap/e2e/cases/*.yaml` for deployed validation coverage.
- Verify each retained local change has a unit test, E2E case, or manual smoke path.

## 2. Porting

- Recreate or cherry-pick local patches onto the new upstream base.
- Prefer replaying the intended behavior when upstream files changed heavily.
- Rebuild the complete image from the root `Dockerfile` after runtime files settle.

## 3. Deployment contract

The runtime command and the proxy config live outside this repository, so
removing a file here can only be caught by checking them by hand.

- When a patch deletes a module, grep the deployment command and the config for
  it. A stale `litellm_settings.callbacks` entry fails the import at startup.
- Verify that the built package contains every new runtime module and generated
  Prisma client required by the migrated commits.

## 4. Validation

- Run unit and integration tests for the changed runtime files.
- Run the local patch E2E suite against the test environment.
- Execute the manual smoke items only for gaps not covered by automation.

## 5. Evidence

- Record the upstream tag, patch branch, and deployed image digest.
- Store E2E results and any failure evidence with timestamps.
- Update image-build and E2E docs when the deployment contract changes.
