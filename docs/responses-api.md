# Qwen Responses, reasoning effort and multimodal API profile

This optional profile fixes API compatibility on the pinned Qwen3.8 Flash-Next
runtime. It layers patch0015 over the existing production-chat-effort profile
(patch0014). Existing production/combined/Chat-only build profiles are unchanged.
No HiCache, kernel, model-weight, sampling, memory-sizing or launch-flag changes
are included. The scheduler delta only reports invalid generated token IDs as
errors instead of successful text termination.

## Behavior

- Responses namespaces retain canonical namespace/member identities on the wire.
  Qwen sees bounded, collision-safe function aliases; request-local inverse maps
  decode them. Forced choices and stored/stateless replay use the same mapping.
  Ordinary/custom tool names keep priority; unknown generated suffixes are not guessed.
- Text streams immediately when tools are present. Intermediate text is commentary;
  final text is final_answer. Phase may be absent while text is streaming and is
  resolved on completion. Split/coalesced markup preserves reasoning/text/tool order.
- Reasoning, commentary and calls from one assistant turn replay together where
  compatible; explicit phase and renewed-reasoning boundaries remain intact.
  Custom tools, tool choice and reasoning replay from the referenced upstream API
  work are backported to the pinned base.
- Qwen effort aliases: minimal→low, high/max→xhigh; low/medium/xhigh remain unchanged.
  none disables thinking. Explicit nested effort wins over top-level effort, which
  wins over the server default. Request-local dictionaries prevent cross-request
  leakage. Chat, Responses and Messages use the common normalization path.
  Patch0014's defensive copies and Chat precedence for other models are retained.
- Responses function/custom tool outputs preserve image parts in their tool turn.
  Missing image detail defaults to auto; SDK iterable outputs are materialized for
  repeatable validation/serialization/replay. Text-only results keep concatenation.
- The release model type enters the existing Qwen video/image processing paths:
  timing metadata and already-sampled-frame markers are forwarded, avoiding a
  second unintended sampling pass; timestamp and image-position handling match
  the existing architecture alias.
- Malformed/unfinished native tool syntax and invalid generated token IDs are
  surfaced as generation errors. Normal model EOS behavior is not replaced by
  automatic retries or an invented final answer.

## Known limitations

**Video ordering is not fixed.** A synthetic four-frame,1fps clip with the true
sequence red→green→blue→yellow can return red→blue→green→yellow, or omit colors.
The decoder and processor audit preserved input order; two-frame temporal packing
is a possible contributor, not a proved exclusive cause. A longer32-frame,4fps
held-color fixture passed, but that does not erase the short-clip failure.
This PR restores metadata/sampling behavior, not guaranteed temporal understanding.

An actual six-round client tool workflow initially attempted to verify an invented
marker, received a tool error, then recovered, completed all six rounds and returned
the final receipt. No routing error or silent stop occurred in that run; the strict
workflow test still failed. Do not interpret passing API tests as perfect model
instruction following or a universal fix for reasoning-only normal-EOS endings.

Token budgets for Messages thinking are not enforced as equivalent external-model
semantics. Hosted tool search/deferred discovery is not implemented by this profile.
The inherited tokenizer may still advertise262144 while an external YaRN2 serving
configuration permits524288; this profile does not alter tokenizer metadata.

## Source profile and provenance

- `patches/0015-qwen-responses-effort-media.patch`: apply after patch0014.
- `runtime-responses-api/python/`: nine changed-file snapshots, separate from
  the existing `runtime/` snapshot so historical profiles stay reproducible.
- `provenance/responses-api-preimages/`: seven exact preimages for offline review.
  The adapter and effort helper are new files.
- `provenance/responses-api.json`: patch and per-file hashes; parent identity.
- `scripts/verify_responses_api.py`: verifies all parent source files before patching
  and all final files afterwards. `--changed-only` is for scoped offline fixtures,
  not full image attestation. Preimage/patch drift fails before application.

The API backport incorporates [upstream38690](https://github.com/sgl-project/sglang/pull/38690)
and [upstream39174](https://github.com/sgl-project/sglang/pull/39174), with pinned-base
adaptation and the Qwen fixes above. These upstream backports include broader
protocol/stream lifecycle changes; the whole patch is not just an alias table.
Existing copyright/license notices and the repository Apache-2.0 license apply.

## Build and run

```bash
docker build -f Dockerfile.responses-api -t sglang-responses-api:local .
```

The Dockerfile pins the published production-chat-effort image by digest and
applies0015 only after full source verification. Use the existing external
[Chat effort launch command](chat-effort-command.sh), substituting the resulting
image. No new serving flags are required. Retain a known-working deployment for
rollback; this PR does not select a GPU memory fraction or KV-token capacity.

## Reproduce CPU checks

No packages are installed by this runner. Use a compatible image already containing
pytest and the pinned serving dependencies, plus local tokenizer/config files:

```bash
python3 scripts/test_responses_api.py --image sglang-responses-api:local --tokenizer /path/to/checkpoint
python3 -m unittest discover -s tests -p 'test*packaging.py' -v
```

For integration testing before building, add `--overlay` to mount just the nine
profile sources over a compatible runtime. The runner is CPU-only, network-disabled,
limited to4CPU/4GiB, and mounts tokenizer data read-only. HF snapshot symlinks retain
access to their sibling blobs directory. Three historical non-Qwen cases remain
explicitly deselected: K2 streaming delimiter, Hunyuan parser and offline Harmony
fixture. No Qwen test is excluded. The existing patch0014 tests are included.

## Validation scope

The integrated CPU suite passed **1081tests and124subtests**, with the3non-Qwen
exclusions above. All7new/existing packaging checks passed. See
`provenance/responses-api-validation.json` for the exact scoped result.
Offline tests reconstruct the patch from preimages, verify all nine snapshots,
reject drift and assert the explicit nine-file scope. CPU tests exercise parsing,
phase ordering, streaming boundaries, namespace collisions/typed arguments,
replay, all three effort protocols, image/tool outputs and video metadata.

Earlier deployed API12 live evidence (before adding the two-file0014 compatibility
integration) includes16/16user/tool-image cases,96/96effort cases across three APIs,
48/48forced catalogue/tool-replay cases, and3/3progressive-streaming checks. The
prior API11 deployment also passed96/96automatic catalogue cases at64clients;
the sampled peak was63active, so this is not proof of sustained64active decode.
A523264-token ordinary retrieval check passed on API10. These are historical,
configuration-specific receipts, not new GPU tests of this PR's integrated build.
Short-video and strict workflow failures above remain explicitly open.

No new GPU deployment, throughput benchmark or full production acceptance is
claimed for this PR. Runtime memory/OOM tuning is outside its scope.
