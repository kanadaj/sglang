"""Packaging contracts for the separate, immutable-parent tool overlay."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_qwen_chat_tools as verifier


def test_exact_order_and_cumulative_inventory():
    manifest, before, after, changed = verifier.records()
    assert manifest['profile'] == 'qwen-chat-tools'
    assert manifest['source_files'] == len(before) == len(after) == 4396
    assert [r['file'] for r in manifest['patches']] == [
        '0048-qwen-composite-schema.patch', '0049-qwen-declared-tool-names.patch',
        '0050-qwen-deferred-reasoning-tools.patch']
    assert len(changed) == 4
    assert before != after


def test_no_unrelated_glm_or_grammar_changes():
    _, before, after, changed = verifier.records()
    assert all('glm' not in p and 'grammar' not in p for p in changed)
    for path in before:
        if 'glm' in path or 'grammar' in path or path.endswith('serving_chat.py'):
            assert before[path] == after[path]


def test_build_has_immutable_parent_and_explicit_revision():
    manifest, _, _, _ = verifier.records()
    dockerfile = (ROOT / 'Dockerfile.qwen-chat-tools').read_text()
    assert 'FROM ' + manifest['base_image'] in dockerfile
    assert 'ARG SOURCE_REVISION' in dockerfile
    assert 'org.opencontainers.image.revision="$SOURCE_REVISION"' in dockerfile
    assert ' --apply' in dockerfile
    assert 'CMD ["--help"]' in dockerfile


def test_runner_fails_closed_and_is_offline():
    script = (ROOT / 'scripts/test_qwen_chat_tools.sh').read_text()
    assert '-euo pipefail -c' in script
    assert '--network none' in script and '--user 1000:1000' in script
    assert '/candidate/tests/runtime_hicache_pr19_embed_tools.py' in script


def test_published_profile_series_remains_unchanged():
    parent = json.loads((ROOT / 'provenance/hicache-pr19-embed-tools.json').read_text())
    assert verifier.digest(ROOT / parent['series_file']) == parent['series_sha256']


def test_actual_tree_intermediate_hash_checks_present():
    code = (ROOT / 'scripts/verify_qwen_chat_tools.py').read_text()
    assert 'Actual patch preimage mismatch' in code
    assert 'Actual patch postimage mismatch' in code
