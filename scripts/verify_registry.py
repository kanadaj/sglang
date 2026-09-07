#!/usr/bin/env python3
"""Read-only anonymous registry audit; no Docker daemon, GPUs or image pull.

Download manifests/config and only the three tiny post-accepted vision layers.
Print public provenance JSON to stdout. Registry bearer tokens stay in memory.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def registry(repo):
    url = 'https://auth.docker.io/token?' + urllib.parse.urlencode({
        'service': 'registry.docker.io', 'scope': f'repository:{repo}:pull'})
    with urllib.request.urlopen(url, timeout=40) as response:
        token = json.load(response)['token']
    def get(kind, ref):
        request = urllib.request.Request(f'https://registry-1.docker.io/v2/{repo}/{kind}/{ref}', headers={
            'Authorization': 'Bearer ' + token,
            'Accept': ', '.join(['application/vnd.oci.image.index.v1+json',
                                'application/vnd.oci.image.manifest.v1+json',
                                'application/vnd.docker.distribution.manifest.list.v2+json',
                                'application/vnd.docker.distribution.manifest.v2+json'])})
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read(5_000_001)
        require(len(data) <= 5_000_000, 'Unexpected large registry object')
        require(ref.startswith('sha256:') and hashlib.sha256(data).hexdigest() == ref[7:], 'Registry digest mismatch')
        return data
    return get

def audit():
    accepted = json.loads((ROOT / 'provenance/build-layers.json').read_text())['layers']
    accepted_digests = [x['digest'] for x in accepted]
    result = {}
    for name, repo, digest in [
        ('day0', 'lmsysorg/sglang', 'sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae'),
        ('vision', 'kanadaj/sglang-qwen38fn-sm120-turbo', 'sha256:8cb9b598ba0be1bbd77924037a517d71e064ef72807b8b96b0fa5c78eeb2f3cc')]:
        get = registry(repo)
        manifest = json.loads(get('manifests', digest))
        index = digest
        if 'manifests' in manifest:
            candidates = [x for x in manifest['manifests'] if x.get('platform', {}).get('architecture') == 'amd64'
                          and x.get('platform', {}).get('os') == 'linux']
            require(len(candidates) == 1, 'Ambiguous Linux amd64 manifest')
            digest = candidates[0]['digest']
            manifest = json.loads(get('manifests', digest))
        get('blobs', manifest['config']['digest'])
        result[name] = dict(repository=repo, index_digest=index, linux_amd64_manifest=digest,
                            config_digest=manifest['config']['digest'], layers=manifest['layers'])
        digests = [x['digest'] for x in manifest['layers']]
        if name == 'day0':
            require(digests == accepted_digests[:len(digests)], 'Day-0 ancestor layer mismatch')
            result[name]['exact_ancestor_layers'] = len(digests)
            continue
        # Docker-v2 vs OCI mediaType conversion does not change compressed bytes.
        require(digests[:len(accepted)] == accepted_digests, 'Vision accepted-parent layer mismatch')
        suffix = manifest['layers'][len(accepted):]
        files = {}
        for layer in suffix:
            require(layer['size'] < 5_000_000, 'Unexpected large vision layer')
            data = get('blobs', layer['digest'])
            require(len(data) == layer['size'], 'Layer size mismatch')
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                for member in archive:
                    if member.isdir():
                        continue
                    require(member.isfile(), 'Unexpected symlink or special file in vision overlay')
                    path = member.name.removeprefix('./')
                    require('..' not in Path(path).parts and '/.wh.' not in path, 'Unexpected layer deletion/traversal')
                    if '__pycache__' in Path(path).parts or path.endswith('.pyc'):
                        continue
                    require(path.startswith('sgl-workspace/sglang/'), 'Unexpected non-source vision modification')
                    rel = path.removeprefix('sgl-workspace/sglang/')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError('Unreadable tar member')
                    content = stream.read()
                    require(content == (ROOT / 'runtime' / rel).read_bytes(), 'Vision overlay content mismatch')
                    files[rel] = hashlib.sha256(content).hexdigest()
        require(set(files) == {'python/sglang/srt/layers/quantization/modelopt_quant.py',
                               'python/sglang/srt/layers/quantization/vision_mxfp8.py'}, 'Unexpected vision source set')
        result[name]['additional_nonbytecode_files'] = files
        result[name]['exact_accepted_parent_layers'] = len(accepted)
    return result

if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
