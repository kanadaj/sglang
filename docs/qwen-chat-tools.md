# Qwen Chat tool reliability overlay (local candidate)

Status: CPU-verified local candidate, **not registry-verified or deployed**. The
README and primary quickstart intentionally retain the existing published release.
After review and merge, rebuild from fetched merged main with its exact Git SHA,
then independently verify registry identity and live GPU behavior before promotion.
No operator launch arguments, templates, model registration, grammar defaults, or
existing profile manifests are changed by this overlay.

## Ordered scope

Immutable cumulative parent:
`docker.io/kanadaj/sglang-qwen38fn-sm120-turbo@sha256:09a132dbfcd2eb4579324c8400e991135aca823a900ee8efe47459daa4931b39`.
The separate `qwen-chat-tools` profile applies:

1. **0048** — exact selected helper and Qwen changes from merged upstream #36626
   (`0665102ce5e2786ec569dc22c017f2f898a94313`): extract top-level properties through
   anyOf/oneOf/allOf. Existing direct properties retain priority. This is type
   conversion, not JSON Schema validation or branch selection. Unrelated detector
   changes in that upstream PR are intentionally excluded.
2. **0049** — adapt upstream #38624 at
   `f3f975f4641e18306e124e46aed7f3c0a39e8bf3`: validate names against declared tools
   before emitting any name/argument deltas. Local 0028 already gates structural
   parsing on genuine wrappers. Unknown complete wrappers are dropped consistently
   in stream/non-stream paths while surrounding normal text is preserved. Only
   original matches are parsed: collecting prose never reparses concatenated spans
   or manufactures a call across a suppressed wrapper. This deliberately differs
   from upstream's streamed literal-name leak. Existing
   no-tools behavior and SGLANG_FORWARD_UNKNOWN_TOOLS opt-in remain supported.
3. **0050** — adapt upstream #42030 at
   `d7f766e60951144e7dfb1b66a859ea246df01428`: defer a tool boundary encountered
   inside Qwen reasoning until explicit reasoning close or EOS. A later close
   keeps the quoted wrapper in reasoning; EOS without that close releases a real
   call to downstream tool parsing. The default shared-detector behavior and GLM
   opt-in remain unchanged. Responses' stream and ordered non-stream adapters must
   enter tool mode only for markers actually released as normal text, not raw
   markers still held by the reasoning parser. Chat already flushes the shared
   ReasoningParser before its downstream function parser; no Chat source change
   was needed. No Anthropic replay backport is included.

Compatibility trade-off: genuine unclosed-think tool calls cannot be emitted
before EOS without reintroducing chunk-dependent phantoms. Such calls are buffered
until final flush. Quoted wrappers similarly delay the reasoning tail until the
close token. This buffering is bounded by generated output length, not a new limit.
Repeated copying/scanning can be quadratic; generation/context limits remain the
operational bound. An unclosed outer reasoning block with a closing-think token
inside a call argument remains ambiguous and conservatively produces no call.
Composite schema branches use first-wins extraction, not discriminated-union
validation. Preexisting non-stream trailing prose after a valid call and Responses
length-termination behavior are not changed by this narrowly scoped overlay.

## Attestation and local reproduction

`provenance/qwen-chat-tools.json` binds each ordered patch hash and actual source
pre/postimages to the cumulative parent inventory. The verifier validates the
parent's predecessor chain, all 4,396 source paths, exact series order, every
applied intermediate file transition, and the full resulting inventory. The full
result inventory is recorded independently in
`provenance/qwen-chat-tools-runtime-files.json`. A fresh immutable-parent extraction
was replayed and verified; applying the series twice and treating the parent image
as the result both fail closed.

For an **uncommitted local candidate**, from this checkout:

```bash
docker build --network none --pull=false \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)-uncommitted-candidate" \
  -f Dockerfile.qwen-chat-tools \
  -t kanadaj/sglang-qwen38fn-sm120-turbo:qwen-chat-tools-20261002-reviewed-candidate .
```

For a **reviewed, clean, fetched merged-main checkout**, the production build must
use `--build-arg SOURCE_REVISION="$(git rev-parse HEAD)"` without the candidate
suffix. No final published tag/digest is claimed here.

The offline CPU runner requires `QWEN_TOKENIZER_PATH` to name an existing resolved
local tokenizer directory. It fails immediately if unset. After setting that
operator-owned path, run:

```bash
bash scripts/test_qwen_chat_tools.sh
```

The candidate tag above is the runner's default; set `QWEN_CHAT_TOOLS_IMAGE` to test
a different already-local rebuilt image. Containers run network-disabled, CPU-only,
non-root, with read-only fixture mounts and fail-fast shell handling.

## Evidence

- Parent RED: schema 48 failed; boundary 8 failed / 8 passed; real ASGI HTTP
  integration 29 failed / 13 passed. The HTTP fixture invokes actual Chat and
  Responses adapters with CPU generation injection and an offline tokenizer; it
  asserts 200 responses, exact content/reasoning/call semantics and parity rather
  than hiding server errors. Fixtures assemble structural markers by concatenation.
- Candidate GREEN: **140** profile regressions/controls, **27** predecessor cumulative
  tests, **17** packaging contracts/negative controls, **190** selected registered
  function-call tests, **2** registered unknown-name tests and **112** full registered
  reasoning tests: **488 unique tests passed**, plus **64** subtests separately.
- Independent splice RED on the earlier local candidate: **1 failed / 3 passed**;
  the reviewed rebuilt candidate passes non-stream and chunks 1, 7 and whole.
- The function-call selection excludes **15** DeepSeek V3.2/V4 cases requiring an
  unavailable offline DeepSeek tokenizer. Running the same unselected registered
  suites on parent and candidate yields the same **15 failed / 304 passed**;
  all 15 failures are missing offline tokenizer assets. This is not a claim that
  the entire registered test tree passed. The full reasoning suite is retained.
- Warnings are existing CPU backend and dependency deprecations, not suppressed
  failures. The concise runner hides warning detail but preserves failure status.

Full logs, source transition manifests, selected upstream diffs and the local
image identity receipt are under `provenance/qwen-chat-tools-*`. GPU validation,
registry publication/readback, final merged revision labeling, and deployment are
separate release gates owned by the release maintainer.
