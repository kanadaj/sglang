"""Offline checks for the pinned cumulative HiCache + tool-grammar release."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_hicache_pr19_embed_tools as release


class ReleaseTests(unittest.TestCase):
    def test_transitions_and_parent_are_audited(self):
        manifest, before, after, changed = release.records()
        self.assertEqual(manifest["base_image"].split("@", 1)[1], "sha256:6acf6306887726b31ad0003de9021fdf0147ffbb1a29a4e3147421bf5063d1e2")
        self.assertEqual(len(before), 4396)
        self.assertEqual(len(after), 4396)
        self.assertEqual([row["file"][:4] for row in manifest["patches"]], ["0046", "0047"])
        self.assertEqual({key for key in before if before[key] != after[key]}, changed)
        for row in manifest["patches"]:
            patch = ROOT / "patches" / row["file"]
            self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), row["sha256"])

    def test_quickstart_uses_published_tool_profile(self):
        text = (ROOT / "docs/quickstart-docker.md").read_text()
        self.assertIn("hicache-pr19-embed-tools-20261001-v1@sha256:09a132dbfcd2eb4579324c8400e991135aca823a900ee8efe47459daa4931b39", text)
        self.assertIn("'context_length'", text)
        self.assertIn("d['grammar_backend']", text)
        self.assertIn("--grammar-backend=none", text)
        self.assertIn("--json-model-override-args", text)
        self.assertNotIn("-e SGLANG_YARN_ROPE_SCALING_FACTOR", text)
        self.assertNotIn("... --enable-hierarchical-cache", text)


if __name__ == "__main__":
    unittest.main()
