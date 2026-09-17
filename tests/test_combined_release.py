"""CPU-only release composition contract; no model or GPU required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CombinedReleaseTest(unittest.TestCase):
    def test_combined_inventory_and_chain(self):
        verifier = ROOT / 'scripts/verify_combined_release.py'
        self.assertTrue(verifier.exists(), 'combined release verifier is missing')
        spec = importlib.util.spec_from_file_location('combined_release', verifier)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest, inventory = module.package_records()
        self.assertEqual(len(inventory), 4394)
        self.assertEqual(len(manifest['files']), 21)
        self.assertEqual(module.inventory_hash(inventory),
                         'bfc9281d655467bbd2bf5620bfb04f2f47a26c86edf1408c809b24ae7903e2cd')
        profiles = [json.loads((ROOT / f'provenance/{p}.json').read_text())
                    for p in ('hicache-wip', 'qwen-strict-tools')]
        expected = sorted([p for profile in profiles for p in profile['patches']],
                          key=lambda p: p['file'])
        self.assertEqual(manifest['patches'], expected)
        result = ROOT / 'provenance/combined-release-runtime-files.json'
        self.assertEqual(json.loads(result.read_text()), inventory)
        self.assertEqual(hashlib.sha256(result.read_bytes()).hexdigest(),
                         manifest['result_inventory_sha256'])


if __name__ == '__main__':
    unittest.main()
