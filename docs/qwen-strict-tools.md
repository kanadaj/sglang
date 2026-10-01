# Qwen tool generation without grammar enforcement

The legacy-named `Dockerfile.qwen-strict-tools` now defaults to unconstrained
output. It no longer applies0032, which promoted omitted tool strictness to true
in Chat and Responses. It applies0028 for Responses streaming stability and0046
for explicit grammar-disabled behavior. Other build profiles keep their defaults.

Why: forcing arbitrary tool schemas through grammar compilation can introduce
long stalls and constrain valid arguments incorrectly. Earlier bounded strict-tool
smoke tests did not establish support for every string pattern or length constraint.
This change avoids that compiler path; it does not repair the compiler or every
XML parser edge case, and it makes no general throughput claim.

## Contract

- The entrypoint selects `--grammar-backend none`.
- Omitted strictness stays omitted; explicit true/false fields are preserved.
- Explicit `strict:true` has no token-mask enforcement in disabled mode.
- Auto/required/named tool choices skip structural and fallback JSON grammars.
- JSON schema, regex, EBNF and structural constraints are accepted without grammar
  compilation, queueing or enforcement. Schema validity checks outside the grammar
  path still apply.
- Tools remain in prompts and normal/streaming tool parsing remains active.
- Schema compliance, required/named selection and a valid tool call are not
  guaranteed. Applications must validate generated arguments before execution.
- An explicit later `--grammar-backend xgrammar` argument re-enables the backend;
  it does not restore strict-by-default normalization. Backend initialization
  failures retain existing error handling instead of silently disabling constraints.

## Build and test

```sh
docker build -f Dockerfile.qwen-strict-tools -t local/qwen-unconstrained .
QWEN_STRICT_IMAGE=local/qwen-unconstrained \
QWEN_TOKENIZER_PATH=/path/to/tokenizer-with-resolved-files \
  bash scripts/test_qwen_strict_tools.sh
```

Legacy Dockerfile, manifest, verifier and test environment names remain compatible.
The pinned parent and complete source inventory are unchanged. Historical0032 is
retained as an unapplied patch for provenance, not included in this profile.
The CPU runner uses no GPU or network and requires existing tokenizer files.
No package installation, cache migration or service restart is performed by tests.

The newly published `hicache-pr19-embed-tools` image contains `0046` plus
`0047`, which makes the historical Qwen strict-by-default wrapper a no-op only
when grammar is disabled. The historical `hicache-pr19-embed` image has neither
patch. The new image retains its normal grammar backend default; use explicit
`--grammar-backend=none` as in the [quickstart](quickstart-docker.md) to opt in.
The flag alone on the older image can abort constrained requests instead of
bypassing grammar work.

## Validation scope

The full4392-file standalone patch chain replays against the exact parent source
inventory. CPU tests cover omitted/false/true strictness, auto/required/named/none
choices, all scheduler constraint fields, enabled-backend controls, Responses
functions/namespaces, real tokenizer tool rendering and streaming parser output.
The local private integration passed Chat, Responses and explicit-strict tool
smoke checks. That deployment includes additional patches; it is not a GPU
qualification of a newly built public image or proof of general tool correctness.
