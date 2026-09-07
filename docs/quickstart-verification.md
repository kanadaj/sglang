# External-runtime quickstart verification (2026-09-07)

Scope: documentation/build packaging correction, not a new serving deployment.
No runtime source patches, checkpoint files, production containers or routes changed.
No new image build or push was required; historical wrapper tags remain intact.

## Registry and installed CLI

- Read back public Docker Hub tag `r22-tp2-vision-pad32-bias-20260907-v2`;
  computed registry-response SHA256 equals its header and the recorded index
  `sha256:8cb9b598ba0be1bbd77924037a517d71e064ef72807b8b96b0fa5c78eeb2f3cc`.
- Re-ran `scripts/verify_registry.py`: Linux/amd64 manifest
  `sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52`,
  exact parent layer chain and vision overlay source bytes verified.
- Inspected the actual installed runtime image Config and historical launcher.
  Entrypoint injects YaRN/context; the documented explicit Python entrypoint
  bypasses it. Config.Env has no model-profile/offline/packed-PLE defaults.
- Ran the exact pinned image with `python3 -m sglang.launch_server --help`:
  exit 0. Runtime warns that the module is supported but `sglang serve` is the
  recommended entrypoint; this is not an argument rejection.
- Extracted the documented command with Bash (Docker replaced by an argv-printing
  shell function), then passed every serving argument to the image's actual
  `ServerArgs.add_cli_args` / argparse parser. Exit 0; TP2, vision, packed offload,
  FP8 E4M3 KV, NEXTN2/3, C16, context 524288 and 0.93 accepted. This tests CLI
  parsing, not `ServerArgs` GPU-dependent initialization or model loading.
- Parsed inline JSON exactly equals the saved full override and the image's
  historical bundled override. The serving command uses only the inline value.
- Fetched the public checkpoint config (no weights) and passed it plus the
  external JSON through the real image's `get_config` / `get_hf_text_config`
  using an ephemeral temporary directory. Resolved context 524288, YaRN factor
  2.0, original context 262144, mRoPE sections [11,11,10], hidden size 2560,
  48 layers, and retained vision configuration. Transformers emitted warnings
  about unrecognized mRoPE keys; fields were retained. No GPU inference claim.

The transient image checks used runc, no GPU device request,
`NVIDIA_VISIBLE_DEVICES=void`, no network or checkpoint/cache mounts,
4 GiB memory / 2 CPU limits and a 90-second timeout. No production restart.

## Repository tests

`PYTHON=<existing CPU-test venv>/bin/python bash scripts/test.sh`:
39 tests passed; clean 11-patch application, 34 changed source hashes and 33
Python syntax checks passed. The two new quickstart tests first failed against
the wrapper packaging, then passed after correction. They verify Bash GPU
quoting, no duplicate serving options, exact deployed flags, complete JSON,
external opt-in environment and build files without packaged launchers.

Not exercised: rebuilding the edited Dockerfiles, online weight download,
a fresh standalone GPU boot, inference examples, long-context accuracy or full
multimodal C16 residency. Existing accepted-runtime evidence is separate.
