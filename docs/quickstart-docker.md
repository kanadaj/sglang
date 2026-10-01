# Quick start — Qwen3.8 Flash Next NVFP4 on two Blackwell GPUs

This is the **published cumulative** image: embedded model overrides, HiCache
hybrid-checkpoint patches `0040`–`0045`, and opt-in tool patches `0046`–`0047`.
The image includes SGLang; you provide the complete local
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` QAD checkpoint. It is not the
unmodified upstream SGLang image, nor the older `embedded-overrides-20260921-v3`
tag. The release manifest, Dockerfile, patch and verifier are in this repository.

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host -p 127.0.0.1:30000:30000 \
  -v /absolute/path/to/Qwen3.8-Flash-Next-NVFP4:/model:ro \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e SGLANG_PLE_PACKED_FP8_REFERENCE=1 \
  -e SGLANG_PRIVATE_DRAFT_NVFP4_A16=1 \
  docker.io/kanadaj/sglang-qwen38fn-sm120-turbo:hicache-pr19-embed-tools-20261001-v1@sha256:09a132dbfcd2eb4579324c8400e991135aca823a900ee8efe47459daa4931b39 \
  --model-path /model \
  --tp-size=2 --quantization=modelopt_mixed \
  --kv-cache-dtype=fp8_e4m3 --context-length=262144 \
  --mem-fraction-static=0.88 --page-size=64 --chunked-prefill-size=4096 \
  --reasoning-parser=auto --tool-call-parser=auto --grammar-backend=none \
  --linear-attn-prefill-backend=flashinfer --linear-attn-decode-backend=flashinfer \
  --mamba-radix-cache-strategy=extra_buffer --mamba-track-interval=128 \
  --mamba-ssm-dtype=bfloat16 --gdn-mtp-cache-mode=none \
  --max-mamba-cache-size=124 --max-running-requests=16 \
  --cuda-graph-max-bs-decode=16 --moe-runner-backend=flashinfer_cutlass \
  --disable-custom-all-reduce --disable-prefill-cuda-graph \
  --ple-offload-embedding \
  --speculative-algorithm=NEXTN --speculative-num-steps=3 \
  --speculative-eagle-topk=1 --speculative-num-draft-tokens=4 \
  --speculative-draft-model-quantization=modelopt_mixed \
  --speculative-moe-runner-backend=flashinfer_cutlass \
  --model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}' \
  --startup-weight-load-mode=serial --mm-enable-dp-encoder \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 30000
```

Choose GPUs with sufficient free VRAM; this command does not reserve them or
move an existing service. The endpoint is
`http://127.0.0.1:30000/v1/chat/completions`. Keep the checkpoint, not just a
config/tokenizer directory, at `/model`. This profile has the embedded
48-layer hybrid/MTP/PLE config at `/opt/qwen-runtime/model_overrides.json`.
The lower `--mem-fraction-static` leaves more vision-encoder workspace than the
older 0.93 setting but reduces KV capacity; it is a starting point, not a
measured concurrency guarantee. `--max-running-requests` and decode graph
coverage are both 16; raising one without the other can fall back to eager
decode. Do not assume a 1M context is validated merely from a successful boot.

## Tool-call behavior

`--grammar-backend=none` **requires patches `0046`–`0047` in the image above**.
It keeps tool definitions in the prompt and keeps Chat/Responses tool parsing,
but bypasses grammar compilation and token masking. Patch `0047` also prevents
the older Qwen strict-by-default wrapper from turning omitted `strict` into
`true` under this mode; explicit client `strict` fields are preserved. This
avoids a known HTTP 400 from some malformed or JavaScript-rounded 64-bit
integer bounds in tool JSON schemas. It does **not** guarantee valid arguments,
`strict:true`, or required or named tool selection. Validate every returned
argument against your own schema **before executing a tool**; handle missing
or malformed calls. If you need grammar-enforced output, omit this flag (or select `xgrammar`), normalize
client tool schemas to safe bounds, and test your actual tools; this patch does
not repair all compiler/schema incompatibilities. See
[tool-call contract](qwen-strict-tools.md).

The running GPUStack production instance may be on an older pinned digest;
publishing this image does not update that instance. Clients on the older
image will not gain patch `0046` by adding the flag alone.

## Extended context (YaRN): explicit CLI override

The embedded config is **native 262144** with no YaRN. To run at 524288,
replace `--context-length=262144` above and add this argument after the image:

```bash
  --context-length=524288 \
  --json-model-override-args '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25},"max_position_embeddings":524288},"max_position_embeddings":524288}'
```

For the experimental 1048576 window use factor `4.0` and
`max_position_embeddings=1048576` at both levels, along with
`--context-length=1048576` and reduced concurrency. These settings are
**deployment arguments**, not baked into the image. See the
[historical production command](production-command.sh) for its explicit factor-4
form. There is **no supported `SGLANG_YARN_ROPE_SCALING_FACTOR` environment
variable in this image**; an earlier quickstart incorrectly recommended it.

## Optional HiCache

For a mounted host directory with adequate RAM and disk budget, add to the
`docker run` options **before the image**:

```bash
  -v /absolute/path/to/cache:/hicache \
  -e SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/hicache \
  -e SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=24G \
```

And add to the SGLang arguments **after the image**:

```bash
  --enable-hierarchical-cache --hicache-write-policy=write_back \
  --hicache-size=8 --hicache-storage-backend=file \
  --hicache-storage-prefetch-policy=wait_complete
```

`--hicache-size=8` is 8 GB **per TP rank** (16 GB total). Size the host and
mount independently; the disk cap also applies per rank. Do not share the
cache namespace between different checkpoints. See the
[HiCache production recipe](hicache-production-recipe.md) for other trade-offs.

## Verify

```bash
curl -fsS http://127.0.0.1:30000/get_server_info | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print('context:', d['context_length'], 'grammar:', d['grammar_backend'])"
```

Also confirm the startup log says `Merged CLI model overrides onto embedded
model config from /opt/qwen-runtime/model_overrides.json`, then test a Chat tool
request with your real schema and validate its returned arguments. The
[release verifier](../scripts/verify_hicache_pr19_embed_tools.py) hashes the
complete source tree and patch transition; a server boot alone is not a tool
correctness test.
