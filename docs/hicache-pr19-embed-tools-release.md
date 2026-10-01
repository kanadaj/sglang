# Cumulative HiCache + tool-call release (2026-10-01)

**Pull:** `docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:hicache-pr19-embed-tools-20261001-v1@sha256:09a132dbfcd2eb4579324c8400e991135aca823a900ee8efe47459daa4931b39` (`linux/amd64`). The tag and index digest were read back anonymously from Docker Hub after push. See the [quickstart](quickstart-docker.md) for a complete launch command and its tool-safety contract.

## Source and reproducibility

- Parent: pinned [HiCache + embedded overrides image](../Dockerfile.hicache-pr19-embed) `sha256:6acf6306887726b31ad0003de9021fdf0147ffbb1a29a4e3147421bf5063d1e2`; its `0040`–`0045` chain and full source inventory are retained unchanged.
- Overlay: ordered [patch series](../patches/series.hicache-pr19-embed-tools): [`0046` grammar-disabled tool handling](../patches/0046-qwen-disable-tool-grammar.patch) and [`0047` Qwen strictness preservation](../patches/0047-qwen-preserve-tool-strictness-without-grammar.patch). The latter affects only explicit grammar-disabled mode; grammar-enabled behavior is unchanged.
- [Dockerfile](../Dockerfile.hicache-pr19-embed-tools), [transition manifest](../provenance/hicache-pr19-embed-tools.json), and [offline verifier](../scripts/verify_hicache_pr19_embed_tools.py) pin both parent and overlay patch hashes and verify all **4,396** SGLang source files before and after application. Three source files change; no model weights or deployment-specific YaRN arguments are baked into this overlay.

```sh
docker build --network none -f Dockerfile.hicache-pr19-embed-tools \
  -t local/sglang-qwen-hicache-tools .
docker run --rm --network none --entrypoint python3 \
  local/sglang-qwen-hicache-tools \
  /opt/hicache-pr19-embed-tools/scripts/verify_hicache_pr19_embed_tools.py \
  --tree /sgl-workspace/sglang
```

## Evidence and limits

The published image passed the full source audit after its offline build, **27** in-image CPU cases covering Chat and Responses tool choices/strictness, disabled/enabled grammar paths, and parser behavior, plus **two new release unit tests** and **two pre-existing standalone quickstart tests**. Its CLI exposes the documented grammar and context flags. The historical parent image fails the new tool-mode tests; the patches are not merely documentation.

**No GPU boot, real-schema tool-call smoke, long-context validation, vision load test, or production rollout of this newly published image was performed.** All four GPUs were occupied at publication time. The running GPUStack model remains on the older `6acf…` image unless independently updated. This release is the newest **published fork profile**, not a claim that its underlying upstream SGLang checkout is the newest upstream release.

With `--grammar-backend=none`, tools remain prompt-visible and parseable but output is not grammar-enforced. Callers must validate arguments and handle missing calls before executing tools. Do not claim that the tool-grammar patches fix every tool-call complaint without testing the complainant's exact image, flags, and schema.
