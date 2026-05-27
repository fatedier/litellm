# Docker Images

TapTap-owned images are built from the repository root `Dockerfile`.

Expected usage:

- build the complete source tree with the root `Dockerfile`
- use an explicit test tag before assigning a release tag
- validate the complete image against the TapTap E2E suite
- do not maintain a selective runtime-file overlay
