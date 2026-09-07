"""Fail-closed packaging checks; no GPU imports."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_source', ROOT / 'scripts/verify_source.py')
assert spec is not None and spec.loader is not None
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

class PackagingTests(unittest.TestCase):
    def test_changed_runtime_hashes_match_manifest(self):
        manifest = json.loads((ROOT / 'provenance/runtime-files.json').read_text())
        expected = sum(row['changed_from_day0'] for row in manifest.values())
        self.assertEqual(verify.verify(ROOT / 'runtime'), expected)
    def test_preimage_drift_is_rejected_before_patching(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / 'tree'
            shutil.copytree(ROOT / 'upstream-preimages', tree)
            path = next(tree.rglob('*.py'))
            path.write_text(path.read_text() + '\n# deliberately corrupted test copy\n')
            with self.assertRaisesRegex(ValueError, 'Preimage mismatch'):
                verify.apply(tree)
    def test_patch_path_coverage_is_exact(self):
        records = json.loads((ROOT / 'provenance/runtime-files.json').read_text())
        patches = json.loads((ROOT / 'provenance/patches.json').read_text())
        declared = {p for patch in patches for p in patch['paths']}
        self.assertEqual(declared, {p for p, row in records.items() if row['changed_from_day0']})
        for patch in patches:
            self.assertEqual(verify.sha(ROOT / 'patches' / patch['file']), patch['sha256'])
    def test_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / 'tree'
            shutil.copytree(ROOT / 'runtime', tree)
            path = next(tree.rglob('*.py'))
            path.write_text(path.read_text() + '\n# deliberately corrupted test copy\n')
            with self.assertRaisesRegex(ValueError, 'Runtime source hash mismatch'):
                verify.verify(tree)
