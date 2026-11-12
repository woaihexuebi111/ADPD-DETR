"""Check downloaded dataset files against the release split manifest."""
import argparse
import json
from pathlib import Path
from dataset_tools import audit_split, resolve_root

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/splits/manifest.json")
    args = p.parse_args()
    reference = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = resolve_root(args.data_root, None)
    for split, expected in reference["splits"].items():
        actual = audit_split(root, split, image_hashes=True)
        indexed = {r["image"]: r for r in actual["files"]}
        wanted = {r["image"]: r for r in expected["files"]}
        missing, extra = wanted.keys() - indexed.keys(), indexed.keys() - wanted.keys()
        changed = [k for k in wanted.keys() & indexed.keys() if indexed[k] != wanted[k]]
        if missing or extra or changed:
            raise RuntimeError(f"{split}: {len(missing)} missing, {len(extra)} extra, {len(changed)} changed files; "
                               f"examples: {list(missing)[:3]}, {list(extra)[:3]}, {changed[:3]}")
        print(f"{split}: verified {actual['images']} images and {actual['instances']} objects")
    print("Dataset matches the release manifest.")


if __name__ == "__main__":
    main()
