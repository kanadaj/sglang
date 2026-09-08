# Packed-PLE HF checkpoint-path correction (2026-09-08)

## Published runtime

```bash
docker pull kanadaj/sglang-qwen38fn-sm120-turbo:r22-tp2-vision-ple-hfpath-20260908-v1@sha256:4c0d09bcf0cb5906e5abe74edf00643d612c7d8f7306ca6e76d38be2d486da58
```

[Standalone command](standalone.md) uses this runtime. All serving arguments and
environment opt-ins remain outside the image. No profile launcher is added;
the entrypoint is `python3 -m sglang.launch_server`, default command `--help`.
The old image's unused historical launcher files remain inherited, not invoked.

## Root cause and source fix

`DefaultModelLoader._prepare_weights` correctly resolves an HF repo ID to the
selected local snapshot, respecting its revision and download directory.
Previously, `_get_all_weights` delayed preparation until tensor iteration and
never passed the resolved identity to Qwen. Packed-PLE's out-of-order
`weight_scale_2` pre-read guessed from config `_name_or_path`, then global server
`model_path`. Both can be HF IDs; the latter can also belong to the target rather
than the draft. GPUStack's local-path launch concealed this defect.

Patch **0012** changes only `model_loader/loader.py` and `models/qwen4_exp.py`:

- Resolve the ordered primary/secondary sources once before `load_weights` and
  reuse those same `ResolvedSource` descriptors for the lazy tensor iterators.
  Resolution errors now surface at iterator construction; tensor reads stay lazy.
- Pass the same descriptors in the already-resolved startup commit path.
- Qwen consumes the model-local descriptor into its load closure at entry,
  including failure paths, rather than mutating config or server globals.
- PLE reads from the primary checkpoint's actual folder/file set; index-selected
  files must belong to that resolved set. Unindexed checkpoints use the selected
  files, not a guessed `model.safetensors` name.
- Keep direct local `load_weights` compatibility. Never perform a second HF
  download/cache search or substitute the target server path. Missing metadata,
  missing keys, nonscalar/nonfinite/nonpositive scales still fail closed.

No packed representation, FP8-reference rounding, TP2 reduction, vision adapter,
quantization kernel or serving flag changed. This patch does not add a new
HF-subfolder option: the pinned loader has no such parameter. Local subdirectories
are tested verbatim. `ShardedStateLoader` is a different pre-sharded state-dict
path and does not call Qwen's checkpoint `load_weights`; remote/streaming loaders
without resolved local safetensors metadata are not newly supported. Packed
storage's existing prohibition on reloading finalized shards remains in force.

## Verification

- Red test on the immutable parent: the **actual imported** loader and Qwen
  `load_weights`, a synthetic offline HF cache with a pinned revision, and two
  indexed safetensors files fail at the reported local-directory error.
- **16 full-module CPU tests** pass in the corrected runtime: HF IDs with config
  and server still IDs; nondefault offline cache; local path/subdirectory;
  interleaved target/draft revisions with different nonunit scales and a stale
  local config path; early resolution with lazy tensor reads; pre-resolved
  commit without another resolution; primary/secondary separation; direct
  local loading; missing index/tensor keys; empty resolved files; invalid scales;
  unselected-revision index tampering; and failure-safe metadata consumption.
- **39 existing CPU tests** pass: packed-PLE interpreter numerics and FP8-reference
  rounding, TP2 rank/reduction seams, vision adapters, source packaging, and
  exact Bash extraction/profile comparison. **55 unittest tests total**, plus
  the actual-image CLI parser and module `--help` checks.
- Clean ordered application reconstructs **35 changed paths** with **34 Python
  files** syntax-checked. The image build verifies **4,390 parent source files**
  before the overlay and all **4,390 corrected installed files** afterward.
- Anonymous registry readback verifies the tag's manifest digest, all **88 parent
  layers**, all **9 added layers**, and the exact two installed SGLang file hashes.
  The other added files are audit metadata/script and licenses, not launchers.
- Independent full-context OMP code review passed. Its first diff-only review
  prompted the load-metadata lifecycle regression; an empty-file concern was
  disproved by the existing unconditional error and an added test.

The synthetic fixture skips GPU model construction and post-load GPU offload,
not the HF resolver, safetensors iterator, scale reader or packed shard loader.
Tests are network-disabled, GPU-free, bounded to one CPU and 3 GiB; the image
build's RUN steps are bounded to one CPU and 2 GiB. Upstream CPU-only AWQ/GGUF,
CUDA-runtime and deprecated module-launch warnings are nonfatal.

**Not verified:** fresh full-checkpoint download, complete standalone GPU boot,
CUDA/NCCL/graph execution, new inference/vision quality or performance. Both
production GPU pairs, routes, model/checkpoint files and stopped experiments
were left untouched. This release is a tested loader correction, not a new
GPU qualification or production rollout.

## Provenance and reproduction

- `Dockerfile.digest`: immutable published vision Linux/amd64 manifest
  `sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52`,
  followed only by the two corrected runtime sources and audits/licenses.
- `patches/series`, `provenance/patches.json`: 12 ordered, hashed patches.
- `upstream-preimages/.../model_loader/loader.py`: newly included clean preimage,
  matched to the accepted export, manifest and immutable parent image.
- `provenance/runtime-files.vision-v2.json`: preserved historical 4,390-file
  inventory; `runtime-files.json`: current corrected inventory. Original source
  identity, deployed manifest, qualification and registry attestations remain
  historical and unchanged. They do not attest patch 0012.
- `provenance/ple-hf-path-{fix,registry,review}.json` and associated logs: new
  release evidence, separate from the original published vision claims.

From a checkout, GPU-free testing of the published image (no source overlays):

```bash
IMAGE=kanadaj/sglang-qwen38fn-sm120-turbo@sha256:4c0d09bcf0cb5906e5abe74edf00643d612c7d8f7306ca6e76d38be2d486da58
docker run --rm --runtime runc --network none --cpus 1 \
  --memory 3g --memory-swap 3g -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD:/tests:ro" --entrypoint bash "$IMAGE" -c \
  'python3 /tests/tests/runtime_checkpoint.py && python3 /tests/tests/runtime_cli.py && cd /tests && bash scripts/test.sh'
```

The 39 legacy CPU tests use the checked-in runtime snapshot; the 16 checkpoint
tests and CLI parser import the image's installed modules directly. The installed
source hash audit ties both to the published bytes.
