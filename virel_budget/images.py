from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from virel_budget.schema import Intervention, Sample


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def materialize_interventions(
    samples: list[Sample],
    intervention_specs: list[dict[str, Any]],
    out_dir: str | Path,
    seed: int,
    reuse_existing: bool = False,
) -> dict[tuple[str, str], Intervention]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    irrelevant_pools = _load_irrelevant_pools(intervention_specs)
    result: dict[tuple[str, str], Intervention] = {}
    for sample in samples:
        original = load_rgb(sample.image_path)
        for spec in intervention_specs:
            name = str(spec["name"])
            kind = str(spec["type"])
            applies_to = spec.get("applies_to")
            if applies_to and sample.dataset not in set(applies_to):
                continue
            variant_path = out / f"{sample.sample_id}_{name}.png"
            if reuse_existing and variant_path.exists() and variant_path.stat().st_size > 0:
                result[(sample.sample_id, name)] = Intervention(
                    name=name,
                    type=kind,
                    path=variant_path,
                    metadata={"spec": spec, "reused_existing": True},
                )
                continue
            if kind == "gray":
                img = Image.new("RGB", original.size, color=(128, 128, 128))
            elif kind == "grayscale":
                img = original.convert("L").convert("RGB")
            elif kind == "black":
                img = Image.new("RGB", original.size, color=(0, 0, 0))
            elif kind == "blur":
                radius = float(spec.get("radius", 8.0))
                img = original.filter(ImageFilter.GaussianBlur(radius=radius))
            elif kind == "counterfactual":
                if sample.counterfactual_image_path is None:
                    continue
                img = load_rgb(sample.counterfactual_image_path).resize(original.size)
            elif kind == "irrelevant":
                img = _choose_irrelevant(sample, original.size, irrelevant_pools.get(name, []), seed)
            else:
                raise ValueError(f"Unknown intervention type: {kind}")
            img.save(variant_path)
            result[(sample.sample_id, name)] = Intervention(
                name=name,
                type=kind,
                path=variant_path,
                metadata={"spec": spec},
            )
    return result


def token_saliency(image_path: str | Path, grid_size: int = 8) -> np.ndarray:
    """Simple CPU saliency proxy: per-cell edge/contrast magnitude."""
    img = load_rgb(image_path).resize((grid_size * 16, grid_size * 16))
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    mag = gx + gy
    h, w = mag.shape
    cells = []
    for y in range(grid_size):
        for x in range(grid_size):
            patch = mag[y * h // grid_size : (y + 1) * h // grid_size, x * w // grid_size : (x + 1) * w // grid_size]
            cells.append(float(patch.mean()))
    return np.asarray(cells, dtype=np.float32)


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_irrelevant_pools(specs: list[dict[str, Any]]) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = {}
    for spec in specs:
        if spec.get("type") != "irrelevant":
            continue
        pool_dir = spec.get("pool")
        if not pool_dir:
            pools[str(spec["name"])] = []
            continue
        directory = Path(pool_dir)
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        pools[str(spec["name"])] = sorted(
            [p for p in directory.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        )
    return pools


def _choose_irrelevant(sample: Sample, size: tuple[int, int], pool: list[Path], seed: int) -> Image.Image:
    if not pool:
        return Image.new("RGB", size, color=(96, 96, 96))
    rng = random.Random(deterministic_seed(seed, sample.sample_id, "irrelevant"))
    candidates = [p for p in pool if p.resolve() != sample.image_path.resolve()]
    chosen = rng.choice(candidates or pool)
    return load_rgb(chosen).resize(size)
