# Docker quickstart: patched SGLang, external serving arguments

Use the original patched vision runtime as ordinary SGLang: every profile flag,
opt-in environment variable and the complete YaRN2 override is supplied below.
No Git checkout, custom launcher, external script or JSON file is required.
The image's historical entrypoint injects YaRN defaults, so **always use the
explicit `--entrypoint python3` and `-m sglang.launch_server` shown here**.

## Requirements

- Linux x86-64, Docker and NVIDIA Container Toolkit, a CUDA 13-compatible driver.
- **Two unused RTX PRO 6000 Blackwell 96 GB GPUs (SM120)** for the documented
  profile. Do not run this on GPUs already serving production. Other cards or
  smaller VRAM are not validated by this recipe.
- Fast local SSD storage for Docker and both named cache volumes. The public
  checkpoint inventory was **105,935,744,072 bytes (~98.66 GiB)** when checked;
  budget at least 150 GiB free for the checkpoint/cache **plus** Docker image
  storage and compilation headroom. Downloads are not bundled in the image.
- Substantial available host RAM: packed PLE alone occupied ~26.82 GiB per TP2
  replica; loading dictionaries, staging, mmap/page cache and engine processes
  need additional RAM. This is **not** a 27 GiB RAM requirement. Peak startup RSS
  and a portable minimum RAM figure have not been established; monitor memory
  and avoid a restrictive container cap or competing loads.

## Pull and run

Run these commands in Bash on your inference host. GPU IDs are host IDs; the
quoted Docker device selector is intentional. The first start downloads
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` from Hugging Face over the network.
The named volumes are created automatically and persist across container removal.
Place Docker's volume storage on local SSD; this recipe does not relocate it.

```bash
IMAGE=kanadaj/sglang-qwen38fn-sm120-turbo:r22-tp2-vision-pad32-bias-20260907-v2@sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52
docker pull "$IMAGE"
docker run -d --name qwen-vision --gpus '"device=0,1"' --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v qwen-vision-cache:/cache \
  -v qwen-vision-kernels:/root/.cache \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e SGLANG_CACHE_DIR=/cache/sglang-generated \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1 \
  -e SGLANG_SM120_ONLINE_MXFP8=false \
  -e SGLANG_PLE_PACKED_NVFP4=1 \
  -e SGLANG_PLE_PACKED_FP8_REFERENCE=1 \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen-vision --host 0.0.0.0 --port 30000 \
  --reasoning-parser=auto \
  --tool-call-parser=auto \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --linear-attn-prefill-backend=flashinfer \
  --linear-attn-decode-backend=flashinfer \
  --max-mamba-cache-size=84 \
  --mamba-radix-cache-strategy=extra_buffer \
  --mamba-track-interval=64 \
  --mamba-ssm-dtype=bfloat16 \
  --gdn-mtp-cache-mode=none \
  --tp-size=2 \
  --quantization=modelopt_mixed \
  --kv-cache-dtype=fp8_e4m3 \
  --context-length=524288 \
  --mem-fraction-static=0.93 \
  --page-size=64 \
  --chunked-prefill-size=4096 \
  --max-running-requests=16 \
  --enable-metrics \
  --enable-cache-report \
  --cuda-graph-max-bs-decode=16 \
  --moe-runner-backend=flashinfer_cutlass \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --ple-offload-embedding \
  --speculative-algorithm=NEXTN \
  --speculative-num-steps=2 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=3 \
  --speculative-draft-model-quantization=modelopt_mixed \
  --speculative-moe-runner-backend=flashinfer_cutlass \
  --model-loader-extra-config '{"enable_multithread_load":false,"num_threads":2}' \
  --startup-weight-load-mode=serial \
  --mm-enable-dp-encoder \
  --json-model-override-args '{"text_config":{"vocab_size":248320,"hidden_size":2560,"intermediate_size":12288,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"hidden_act":"silu","max_position_embeddings":524288,"initializer_range":0.02,"rms_norm_eps":1e-06,"use_cache":true,"head_dim":256,"attention_bias":false,"attention_dropout":0.0,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":48,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"num_experts_per_tok":10,"num_experts":512,"output_router_logits":false,"router_aux_loss_coef":0.001,"hc_count":4,"hc_lowrank":320,"ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":128,"output_gate_type":"sigmoid","indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25},"output_hidden_states":false,"return_dict":true,"dtype":"bfloat16","chunk_size_feed_forward":0,"is_encoder_decoder":false,"id2label":{"0":"LABEL_0","1":"LABEL_1"},"label2id":{"LABEL_0":0,"LABEL_1":1},"problem_type":null,"_name_or_path":"","pad_token_id":null,"bos_token_id":248044,"eos_token_id":248044,"tie_word_embeddings":false,"mamba_ssm_dtype":"float32","mtp":{"hybrid":true,"layer_types":["full_attention"],"mtp_use_hidden_state_from_layer":null,"num_hidden_layers":1,"rope_theta":10000000},"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"model_type":"qwen3_8_flash_next_text","output_attentions":false,"ple_embedding_dtype":"nvfp4"},"max_position_embeddings":524288}'
```

```bash
docker logs -f qwen-vision
# In another terminal; wait for loading/compilation before expecting HTTP 200:
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/v1/models
curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-vision","messages":[{"role":"user","content":"Reply with hello."}],"max_tokens":128}'
```

### Image request (Python standard library only)

This creates a solid red PNG in memory; no file, package install or repo checkout
is needed. Inspect the answer for red; HTTP 200 alone is not accuracy validation.

```bash
python3 - <<'PY'
import base64, json, struct, urllib.request, zlib

def chunk(kind, data):
    return (struct.pack('>I', len(data)) + kind + data
            + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))

png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress((b'\x00' + b'\xff\x00\x00' * 224) * 224))
       + chunk(b'IEND', b''))
payload = {"model": "qwen-vision", "max_tokens": 512, "messages": [{
    "role": "user", "content": [
        {"type": "text", "text": "What single color fills this image?"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]}
request = urllib.request.Request(
    'http://127.0.0.1:30000/v1/chat/completions',
    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

### Customize the ordinary SGLang command

Edit flags in place, rather than depending on injected defaults or duplicate
options. For existing **complete matching** local weights, add
`-v /absolute/path/to/checkpoint:/model:ro` before `--entrypoint`, and replace
`--model-path` with `/model`. Include vision processor/tokenizer files.
No cached checkpoint configuration is modified. Changing context length alone
does not remove YaRN: edit/remove the explicit JSON override as appropriate.
The complete `text_config` intentionally preserves the saved deployed profile.
The runtime updates existing config objects one level deep; it does not recursively
merge arbitrary nested dictionaries. In particular, supply the complete
`rope_parameters`, including mRoPE fields, rather than only the YaRN factor.

The API listens on all interfaces inside the container but is published only
on host loopback. For deliberate LAN access, change the host bind address and
apply firewall/authentication controls; do not expose this unauthenticated API
to the Internet. Host IPC reduces isolation and is for trusted workloads.
Weights download on first use; the two named volumes preserve weights/generated
files and compilation caches. All performance opt-ins are external `-e` options.
The actual runtime image Config.Env contains CUDA/build/library settings, but
no model-specific serving profile, offline, packed-PLE or YaRN environment defaults.
Its inherited `NVIDIA_VISIBLE_DEVICES=all` is constrained by the explicit Docker
GPU device request. Inherited Entrypoint is
`["/opt/nvidia/nvidia_entrypoint.sh","python3","/opt/qwen-yarn/launch.py"]`,
Cmd is `["--help"]`; the command above replaces both and never reads bundled YaRN.

## Publication and verification boundary

See [verification details](quickstart-verification.md) for exact test scope,
config-resolution results and non-fatal upstream CLI/mRoPE warnings.

- Original vision index: `sha256:8cb9b598ba0be1bbd77924037a517d71e064ef72807b8b96b0fa5c78eeb2f3cc`.
- Pinned Linux/amd64 manifest used above: `sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52`.
- The former `r22-tp2-vision-standalone-20260907-v1` convenience wrapper is
  **superseded**, not the recommended image. Its remote tag/history are retained;
  no new wrapper image is needed or published for this correction.
- Tests extract this command with Bash and compare flags/environment/complete
  JSON to the saved profile. Exact-image module `--help` and real CLI parser
  checks are GPU-free, network-disabled, with no checkpoint mounts.
- **Not tested here:** a fresh online checkpoint download, standalone GPU boot,
  or the HTTP examples against a new standalone server. Production was not
  restarted or changed. See [runtime evidence and caveats](../README.md#evidence-and-known-limits).
  YaRN2 is extrapolation; 524K multimodal accuracy/full C16 residency is not
  guaranteed. Parser acceptance is not end-to-end GPU qualification.
