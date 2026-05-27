# TapTap Local Extensions

This directory contains TapTap-owned assets that should stay clearly separated
from upstream LiteLLM code during upgrades.

Scope:

- patch-specific Docker overlay files
- E2E test specifications and helper scripts
- upgrade runbooks and validation checklists

Guidelines:

- Keep runtime code changes in their existing import locations unless there is a
  strong reason to move them.
- Keep Docker overlay and E2E coverage notes here instead of scattering them
  across the repository root.
- Treat the Docker overlay and E2E cases as the source of truth for upgrade
  planning and post-deploy validation.
