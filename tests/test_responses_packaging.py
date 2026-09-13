"""Offline patch scope, reconstruction and drift rejection for Responses."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_responses_api", ROOT / "scripts/verify_responses_api.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ResponsesPackagingTests(unittest.TestCase):
    def test_scope_and_snapshot_hashes(self):
        records = json.loads((ROOT / "provenance/responses-api.json").read_text())["files"]
        expected = {
            "entrypoints/harmony_utils.py", "entrypoints/openai/protocol.py",
            "entrypoints/openai/serving_responses.py", "entrypoints/openai/responses_adapters.py",
            "entrypoints/openai/serving_chat.py", "entrypoints/openai/qwen_effort.py",
            "function_call/qwen3_coder_detector.py", "managers/schedule_batch.py",
            "multimodal/processors/qwen_vl.py",
        }
        self.assertEqual(set(records), {"python/sglang/srt/" + p for p in expected})
        self.assertEqual(verify.verify(ROOT / "runtime-responses-api", changed_only=True), 9)
        self.assertEqual(len(verify.inventory(True)), len(verify.inventory(False)) + 2)

    def test_clean_apply_matches_all_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            shutil.copytree(ROOT / "provenance/responses-api-preimages", tree)
            self.assertEqual(verify.apply(tree, changed_only=True), 9)
            for p in (ROOT / "runtime-responses-api").rglob("*.py"):
                rel = p.relative_to(ROOT / "runtime-responses-api")
                self.assertEqual((tree / rel).read_bytes(), p.read_bytes())
                compile(p.read_bytes(), str(rel), "exec")

    def test_preimage_drift_fails_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            shutil.copytree(ROOT / "provenance/responses-api-preimages", tree)
            p = tree / "python/sglang/srt/entrypoints/openai/protocol.py"
            p.write_text(p.read_text() + "\n# changed fixture\n")
            with self.assertRaisesRegex(ValueError, "Source mismatch"):
                verify.apply(tree, changed_only=True)
            self.assertFalse((tree / "python/sglang/srt/entrypoints/openai/responses_adapters.py").exists())
