"""Independent packaging tamper and shell-control regressions."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from test_qwen_chat_tools_packaging import ROOT, verifier


@pytest.mark.parametrize('target', ['series', 'patch', 'parent', 'before', 'after', 'inventory', 'count'])
def test_rejects_packaging_tamper(tmp_path, monkeypatch, target):
    for directory in ['patches', 'provenance']:
        shutil.copytree(ROOT / directory, tmp_path / directory)
    manifest_path = tmp_path / 'provenance/qwen-chat-tools.json'
    manifest = json.loads(manifest_path.read_text())
    if target == 'series':
        path = tmp_path / manifest['series_file']
        path.write_text('\n'.join(reversed(path.read_text().splitlines())) + '\n')
    elif target == 'patch':
        path = tmp_path / 'patches' / manifest['patches'][1]['file']
        path.write_text(path.read_text() + '\n# tampered\n')
    elif target == 'parent':
        manifest['parent_inventory_sha256'] = '0' * 64
    elif target in ['before', 'after']:
        hashes = next(iter(manifest['patches'][1]['files'].values()))
        hashes[target] = '0' * 64
    elif target == 'inventory':
        path = tmp_path / 'provenance/qwen-chat-tools-runtime-files.json'
        inventory = json.loads(path.read_text())
        inventory.pop(next(iter(inventory)))
        path.write_text(json.dumps(inventory))
    else:
        manifest['source_files'] += 1
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        verifier.records()


def test_docker_copy_dependencies_and_fixed_base():
    dockerfile = (ROOT / 'Dockerfile.qwen-chat-tools').read_text()
    assert 'FROM docker.io/kanadaj/sglang-qwen38fn-sm120-turbo@sha256:09a132dbfcd2eb4579324c8400e991135aca823a900ee8efe47459daa4931b39\n' in dockerfile
    copied = set()
    for line in dockerfile.splitlines():
        if line.startswith('COPY '):
            for source in line.split()[1:-1]:
                assert (ROOT / source).exists()
                copied.add(source)
    assert {'scripts/verify_qwen_chat_tools.py', 'scripts/verify_hicache_pr19_embed.py',
            'scripts/verify_hicache_pr19_embed_tools.py', 'provenance/qwen-chat-tools.json',
            'provenance/qwen-chat-tools-runtime-files.json',
            'provenance/hicache-pr19-embed-runtime-files.json',
            'provenance/hicache-pr19-embed.json', 'provenance/hicache-pr19-embed-tools.json',
            'patches/'} <= copied
    assert 'org.opencontainers.image.description=' in dockerfile


def test_inner_shell_stops_at_verifier_failure():
    script = (ROOT / 'scripts/test_qwen_chat_tools.sh').read_text()
    body = script.split("-euo pipefail -c '\n", 1)[1].rsplit("'", 1)[0]
    prefix = 'python3() { return 42; }; export -f python3;\n'
    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', prefix + body], capture_output=True)
    assert result.returncode == 42


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_actual_intermediate_hash_mismatch_stops_apply(tmp_path, monkeypatch, phase):
    path = tmp_path / 'file.py'
    path.write_bytes(b'original')
    before = verifier.digest(path)
    path.write_bytes(b'changed')
    after = verifier.digest(path)
    path.write_bytes(b'original' if phase == 'after' else b'tampered')
    manifest = {'patches': [{'file': 'patch', 'files': {'file.py': {'before': before, 'after': after}}}]}
    monkeypatch.setattr(verifier, 'records', lambda: (manifest, {}, {}, set()))
    monkeypatch.setattr(verifier, 'verify_tree', lambda *args: None)
    applied = []
    monkeypatch.setattr(verifier, 'apply_patch', lambda *args: applied.append(True))
    with pytest.raises(ValueError, match='Actual patch ' + ('preimage' if phase == 'before' else 'postimage')):
        verifier.verify(tmp_path, apply=True)
    assert bool(applied) == (phase == 'after')
