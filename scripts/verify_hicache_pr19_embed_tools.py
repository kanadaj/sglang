#!/usr/bin/env python3
"""Verify the pinned PR#19 image plus opt-in, unconstrained tool-grammar patch."""
import argparse
import json
from pathlib import Path

from verify_hicache_pr19_embed import (
    ROOT, apply_patch, digest, inventory_hash, package_records, verify_tree,
)


def records():
    manifest = json.loads((ROOT / "provenance/hicache-pr19-embed-tools.json").read_text())
    parent, inventory = package_records()
    if manifest["parent_inventory_sha256"] != parent["result_inventory_sha256"]:
        raise ValueError("Parent source inventory digest mismatch")
    series = ROOT / manifest["series_file"]
    if digest(series) != manifest["series_sha256"] or series.read_text().splitlines() != [
        row["file"] for row in manifest["patches"]
    ]:
        raise ValueError("Tool patch series order or digest mismatch")
    before = dict(inventory)
    changed = set()
    for patch_record in manifest["patches"]:
        patch = ROOT / "patches" / patch_record["file"]
        if digest(patch) != patch_record["sha256"]:
            raise ValueError("Tool patch digest mismatch: " + patch.name)
        for name, hashes in patch_record["files"].items():
            if inventory.get(name) != hashes["before"] or hashes["after"] == hashes["before"]:
                raise ValueError("Tool source transition mismatch: " + name)
            inventory[name] = hashes["after"]
            changed.add(name)
    if len(inventory) != manifest["source_files"] or inventory_hash(inventory) != manifest["result_inventory_sha256"]:
        raise ValueError("Tool patches resulting source inventory mismatch")
    return manifest, before, inventory, changed


def verify(tree: Path, apply: bool = False):
    manifest, before, after, changed = records()
    if apply:
        verify_tree(tree, before)
        for patch_record in manifest["patches"]:
            apply_patch(tree, ROOT / "patches" / patch_record["file"])
    verify_tree(tree, after)
    return {"profile": manifest["profile"], "parent": manifest["parent_profile"],
            "source_files": len(after), "changed_source_files": len(changed),
            "full_tree_verified": True, "clean_patch_apply": apply}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and args.tree is None:
        parser.error("--apply requires --tree")
    if args.tree is None:
        manifest, _, after, changed = records()
        result = {"profile": manifest["profile"], "source_files": len(after),
                  "changed_source_files": len(changed),
                  "full_tree_verified": False, "clean_patch_apply": False}
    else:
        result = verify(args.tree.resolve(), args.apply)
    print(json.dumps(result))
