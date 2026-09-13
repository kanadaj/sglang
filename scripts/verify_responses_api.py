#!/usr/bin/env python3
"""Reconstruct and verify the opt-in Responses profile without GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/responses-api.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(final):
    records = json.loads((ROOT / "provenance/production/runtime-files.json").read_text())
    expected = {path: row["sha256"] for path, row in records.items()}
    chat = json.loads((ROOT / "provenance/chat-effort.json").read_text())["files"]
    for path, row in chat.items():
        if expected[path] != row["before"]:
            raise ValueError("Chat preimage mismatch: " + path)
        expected[path] = row["after"]
    if final:
        for path, row in json.loads(MANIFEST.read_text())["files"].items():
            if expected.get(path) != row["before"]:
                raise ValueError("Responses preimage mismatch: " + path)
            expected[path] = row["after"]
    return expected


def verify(tree, *, final=True, changed_only=False):
    expected = inventory(final)
    if changed_only:
        changed = json.loads(MANIFEST.read_text())["files"]
        expected = {p: expected[p] for p in changed if p in expected}
    else:
        actual = {str(p.relative_to(tree)) for p in (tree / "python/sglang").rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if actual != set(expected):
            raise ValueError("Full source inventory differs")
    for path, digest in expected.items():
        if sha(tree / path) != digest:
            raise ValueError("Source mismatch: " + path)
    return len(expected)


def apply(tree, *, changed_only=False):
    manifest = json.loads(MANIFEST.read_text())
    patch = ROOT / "patches" / manifest["patch"]
    if sha(patch) != manifest["patch_sha256"]:
        raise ValueError("Patch digest mismatch")
    verify(tree, final=False, changed_only=changed_only)
    for path, row in manifest["files"].items():
        if row["before"] is None and (tree / path).exists():
            raise ValueError("New path already exists: " + path)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
    return verify(tree, changed_only=changed_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--changed-only", action="store_true", help="Scoped offline fixtures only; not a full image attestation")
    args = parser.parse_args()
    tree = args.tree.resolve()
    count = (apply if args.apply else verify)(tree, changed_only=args.changed_only)
    print(json.dumps({"profile": "production-responses-api", "verified_source_files": count,
                      "full_inventory": not args.changed_only}))
