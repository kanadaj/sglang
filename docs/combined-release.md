# Combined PR15 + PR16 publication

This opt-in, runtime-only profile composes the complete HiCache series (0020–0027,
0029–0031) and the strict-tools series (0028, 0032) in numerical order on their
common digest-pinned parent. It does not change the existing separate profiles.
The source revision is merged main `a6d5284446b539f7ed95ddad9dfc2820f5778e2a`.
The OCI image revision is the exact committed packaging revision used for the
build. The merged source base is recorded separately in the manifest and the
`io.github.kanadaj.sglang.source-base-revision` image label.

`provenance/combined-release.json` records every patch SHA256 and before/after
transition. `provenance/combined-release-runtime-files.json` inventories all 4394
result source files; the parent inventory includes all 4392 original files.
The verifier checks the complete parent tree before applying patches, then checks
the complete result tree, not merely the changed paths. No source overlays are
used when testing the built image. Python bytecode caches are removed on build.

Build with an unused descriptive tag (never overwrite production/latest):

```sh
docker build --network none --pull=false -f Dockerfile.combined-release \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t "$COMBINED_IMAGE" .
COMBINED_IMAGE="$COMBINED_IMAGE" QWEN_TOKENIZER_PATH=/path/to/resolved/tokenizer \
  bash scripts/test_combined_release.sh
```

CPU test containers use runc (no GPU runtime), no network, non-root, read-only
filesystems, resource limits, and no capabilities. Model launch arguments, cache
configuration, endpoint routing, and fleet rollout remain external. The default
command is `--help`; this publication neither registers nor deploys the image.

HiCache remains experimental: the inherited profile documents unresolved CUDA
graph/NaN and tool-replay issues. CPU passing does not qualify GPU kernels,
TP/distributed behavior, numerical accuracy, live serving, or cache enablement.
Do not infer production readiness from publication.

After push, compare the registry manifest digest with the push result and local
RepoDigests. Separately compare the remote manifest config.digest with the local
configuration digest. Classic Docker exposes that as image .Id, but Docker 29's
containerd backend can expose the OCI index digest there instead. On that backend,
read and hash the local runtime manifest/configuration with containerd's content
API. For OCI indexes, select the linux/amd64 runtime manifest, not an attestation.
These are different identities and must not be conflated.
