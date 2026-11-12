"""Dataset validation and portable fingerprints for the ComplexSea release."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_root(data_root: Path | None, data_yaml: Path | None) -> Path:
    if data_root is not None:
        root = data_root.expanduser().resolve()
    else:
        import yaml
        cfg_path = data_yaml.expanduser().resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
        for split in ("train", "val", "test"):
            if cfg.get(split) != f"images/{split}":
                raise ValueError("Use the supplied ComplexSea directory layout or --data-root")
        root = Path(cfg["path"]).expanduser()
        if not root.is_absolute():
            root = cfg_path.parent / root
        root = root.resolve()
    if not (root / "images").is_dir() or not (root / "labels").is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {root}")
    return root


def audit_split(root: Path, split: str, image_hashes: bool = False) -> dict:
    images = sorted(p for p in (root / "images" / split).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    records = []
    seen = set()
    for image in images:
        relative = image.relative_to(root / "images" / split)
        if image.stem in seen:
            raise ValueError(f"Duplicate image stem: {image.stem}")
        seen.add(image.stem)
        label = root / "labels" / split / relative.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing label file: {label}; use an empty file for a true negative image")
        instances = 0
        for lineno, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError(f"Invalid YOLO row at {label}:{lineno}")
            cls, cx, cy, width, height = values
            if cls != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid class/normalized coordinates at {label}:{lineno}")
            instances += 1
        record = {"image": (Path("images") / split / relative).as_posix(),
                  "label": label.relative_to(root).as_posix(), "instances": instances,
                  "label_sha256": sha256(label)}
        if image_hashes:
            record["image_sha256"] = sha256(image)
        records.append(record)
    fingerprint = "\n".join(f"{r['label']} {r['label_sha256']}" for r in records).encode()
    return {"split": split, "images": len(records), "instances": sum(r["instances"] for r in records),
            "labels_sha256": hashlib.sha256(fingerprint).hexdigest(), "files": records}
