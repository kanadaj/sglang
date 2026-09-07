# Standalone Docker quickstart (no clone, no GPUStack)

This published convenience image bundles the complete TP2 vision profile and
YaRN2 JSON. It adds only launch/config/license files to the immutable published
vision runtime; the original vision tag and production deployments are unchanged.

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
IMAGE=kanadaj/sglang-qwen38fn-sm120-turbo:r22-tp2-vision-standalone-20260907-v1@sha256:853293da47f2e968fec680952d682891305e943d36b6aa7863d85fa5c939f65f
docker pull "$IMAGE"
docker run -d --name qwen-vision --gpus '"device=0,1"' --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v qwen-vision-cache:/cache \
  -v qwen-vision-kernels:/root/.cache \
  "$IMAGE"
```

The API listens on all interfaces **inside** the container but Docker publishes
it only on host loopback. For deliberate LAN access, replace the published host
address with your host's LAN IP and add firewall/access controls. Do not expose
this unauthenticated endpoint to the public Internet. Host IPC is for the trusted
multi-GPU workload and reduces IPC isolation.

Defaults: TP2, vision DP encoder, packed CPU PLE with FP8-reference rounding,
`modelopt_mixed`, FP8 E4M3 KV, NEXTN 2 steps / 3 draft tokens, C16, context 524288,
YaRN2 and memory fraction 0.93. No `--language-model-only` or offline flags are
injected. Weights are downloaded by SGLang/Hugging Face, not by this repository.
The cache includes HF weights and SGLang-generated files under `/cache`, plus
library compilation caches under `/root/.cache`.

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

### Existing local weights and overrides

For an existing **complete matching** checkpoint (including processor/tokenizer),
add `-v /absolute/path/to/checkpoint:/model:ro` before `"$IMAGE"` and append
`--model-path /model` after it. No cached checkpoint JSON is modified. Named
compilation caches may still be useful. No separate YaRN file is needed.

Additional canonical long flags after the image override matching defaults,
for example `--max-running-requests 8`. Both `--flag value` and `--flag=value`
work; use canonical names rather than aliases. Environment defaults can be
replaced with Docker `-e NAME=value`. `--help` prints the actual SGLang CLI.
Changing context length alone does not remove the bundled YaRN2 override.
Do not add language-only mode when you need vision. To remove/restructure an
entire profile rather than overriding a value, use the base image and explicit
arguments instead.

## Publication and verification boundary

- Standalone OCI index: `sha256:853293da47f2e968fec680952d682891305e943d36b6aa7863d85fa5c939f65f`.
- Linux/amd64 manifest: `sha256:cb6fa39f4bde4931f6e9d20fcfda31c9945c5e9d95231191578c49c9db30435a`.
- Runtime parent: `sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52`.
- Built with `Dockerfile.standalone`, pushed, and independently read back from
  Docker Hub. All parent layer digests were preserved; downloaded added files
  matched this repository. No original tag was overwritten.
- GPU-free image tests exercised bundled argv/environment/defaults/overrides,
  actual `ServerArgs` CLI parsing, and default-entrypoint `--help`. No GPUs,
  checkpoint mounts or network were available to these smoke containers.
- **Not tested here:** a fresh online checkpoint download, standalone GPU boot,
  or these HTTP examples against a new standalone server. All four production
  GPUs were left occupied by their existing deployments; no competing server
  was launched. The parent runtime's separate vision evidence and OCR/latency
  caveats remain in [README](../README.md#evidence-and-known-limits).
  YaRN2 is extrapolation; 524K multimodal accuracy/full C16 residency is not
  guaranteed. This is a verified publication/configuration recipe, not a new
  end-to-end GPU qualification.
