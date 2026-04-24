#!/usr/bin/env python3
"""Shared config and pattern utilities for ISP-CSIIR flows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PATTERN_GENERATORS = {
    "zeros": lambda h, w, seed=None: np.zeros((h, w), dtype=np.int32),
    "ramp": lambda h, w, seed=None: np.fromfunction(
        lambda j, i: (i + j) % 1024, (h, w), dtype=np.int32
    ),
    "random": lambda h, w, seed=None: np.random.RandomState(seed).randint(
        0, 1024, (h, w), dtype=np.int32
    ),
    "checkerboard": lambda h, w, seed=None: np.fromfunction(
        lambda j, i: ((i // 8) + (j // 8)) % 2 * 1023, (h, w), dtype=np.int32
    ),
    "max": lambda h, w, seed=None: np.full((h, w), 1023, dtype=np.int32),
    "gradient": lambda h, w, seed=None: np.fromfunction(
        lambda j, i: (i * 4) % 1024, (h, w), dtype=np.int32
    ),
}


@dataclass(frozen=True)
class RegisterSpec:
    reg_address: str
    reg_name: str
    bitwidth: int
    initial_value: int
    cons_min: int
    cons_max: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _repo_root() / path


def load_json(path_like: str | Path) -> Dict[str, Any]:
    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path_like: str | Path, payload: Dict[str, Any]) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def load_register_table(csv_path: str | Path) -> List[RegisterSpec]:
    path = resolve_repo_path(csv_path)
    specs: List[RegisterSpec] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            specs.append(
                RegisterSpec(
                    reg_address=row["reg_address"],
                    reg_name=row["reg_name"],
                    bitwidth=int(row["bitwidth"]),
                    initial_value=int(row["initial_value"]),
                    cons_min=int(row["cons_min"]),
                    cons_max=int(row["cons_max"]),
                )
            )
    return specs


def validate_register_values(specs: Iterable[RegisterSpec], regs: Dict[str, int]) -> Dict[str, int]:
    validated: Dict[str, int] = {}
    for spec in specs:
        value = int(regs.get(spec.reg_name, spec.initial_value))
        if not spec.cons_min <= value <= spec.cons_max:
            raise ValueError(
                f"{spec.reg_name}={value} out of range [{spec.cons_min}, {spec.cons_max}]"
            )
        validated[spec.reg_name] = value
    return validated


def generate_reg_config(csv_path: str | Path) -> Dict[str, int]:
    specs = load_register_table(csv_path)
    return validate_register_values(specs, {})


def load_reg_config(json_path: str | Path, csv_path: str | Path) -> Dict[str, int]:
    specs = load_register_table(csv_path)
    raw = load_json(json_path)
    return validate_register_values(specs, {key: int(value) for key, value in raw.items()})


def reg_config_to_model_config(regs: Dict[str, int]) -> Dict[str, Any]:
    return {
        "IMG_WIDTH": regs["reg_image_width"],
        "IMG_HEIGHT": regs["reg_image_height"],
        "win_size_thresh": [
            regs["reg_win_size_thresh_0"],
            regs["reg_win_size_thresh_1"],
            regs["reg_win_size_thresh_2"],
            regs["reg_win_size_thresh_3"],
        ],
        "win_size_clip_y": [
            regs["reg_win_size_clip_y_0"],
            regs["reg_win_size_clip_y_1"],
            regs["reg_win_size_clip_y_2"],
            regs["reg_win_size_clip_y_3"],
        ],
        "win_size_clip_sft": [
            regs["reg_win_size_clip_sft_0"],
            regs["reg_win_size_clip_sft_1"],
            regs["reg_win_size_clip_sft_2"],
            regs["reg_win_size_clip_sft_3"],
        ],
        "blending_ratio": [
            regs["reg_blending_ratio_0"],
            regs["reg_blending_ratio_1"],
            regs["reg_blending_ratio_2"],
            regs["reg_blending_ratio_3"],
        ],
        "reg_edge_protect": regs["reg_edge_protect"],
    }


def load_img_info(json_path: str | Path) -> Dict[str, Any]:
    payload = load_json(json_path)
    image = payload.setdefault("image", {})
    outputs = payload.setdefault("outputs", {})
    payload.setdefault("seed", 0)
    image.setdefault("width", 64)
    image.setdefault("height", 64)
    image.setdefault("bitwidth", 10)
    image.setdefault("source", "generate")
    image.setdefault("pattern", "random")
    image.setdefault("input_path", f"patterns/input/csiir_input_seed{payload['seed']}.hex")
    outputs.setdefault("py_pattern_path", f"patterns/output/csiir_output_py_seed{payload['seed']}.hex")
    outputs.setdefault("hls_pattern_path", f"patterns/output/csiir_output_hls_seed{payload['seed']}.hex")
    payload.setdefault("generator", {})
    payload["generator"].setdefault("force_regenerate", False)
    return payload


def load_hex_image(path_like: str | Path, width: int, height: int) -> np.ndarray:
    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        values = [int(line.strip(), 16) for line in handle if line.strip()]
    expected = width * height
    if len(values) != expected:
        raise ValueError(f"{path} contains {len(values)} samples, expected {expected}")
    return np.array(values, dtype=np.int32).reshape(height, width)


def save_hex_image(path_like: str | Path, image: np.ndarray) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in image.reshape(-1):
            handle.write(f"{int(value) & 0x3ff:03x}\n")
    return path


def generate_input_pattern(img_info: Dict[str, Any]) -> np.ndarray:
    image = img_info["image"]
    width = int(image["width"])
    height = int(image["height"])
    pattern = image["pattern"]
    seed = img_info.get("seed")
    if pattern not in PATTERN_GENERATORS:
        raise ValueError(f"unsupported pattern type: {pattern}")
    return PATTERN_GENERATORS[pattern](height, width, seed=seed)


def prepare_input_image(img_info: Dict[str, Any]) -> Tuple[np.ndarray, Path]:
    image = img_info["image"]
    path = resolve_repo_path(image["input_path"])
    source = image["source"]
    width = int(image["width"])
    height = int(image["height"])
    force_regenerate = bool(img_info.get("generator", {}).get("force_regenerate", False))

    if source == "file":
        return load_hex_image(path, width, height), path

    if source != "generate":
        raise ValueError(f"unsupported image source: {source}")

    if path.exists() and not force_regenerate:
        return load_hex_image(path, width, height), path

    image_data = generate_input_pattern(img_info)
    save_hex_image(path, image_data)
    return image_data, path


def compare_hex_images(lhs_path: str | Path, rhs_path: str | Path, width: int, height: int) -> Dict[str, Any]:
    lhs = load_hex_image(lhs_path, width, height)
    rhs = load_hex_image(rhs_path, width, height)
    diff = lhs.astype(np.int32) - rhs.astype(np.int32)
    diff_mask = diff != 0
    mismatch_indices = np.argwhere(diff_mask)
    mismatches = [
        {
            "y": int(y),
            "x": int(x),
            "lhs": int(lhs[y, x]),
            "rhs": int(rhs[y, x]),
            "delta": int(diff[y, x]),
        }
        for y, x in mismatch_indices[:32]
    ]
    return {
        "total_pixels": int(width * height),
        "mismatch_count": int(diff_mask.sum()),
        "max_abs_diff": int(np.abs(diff).max(initial=0)),
        "mismatches": mismatches,
    }


def normalize_pattern_role(path_like: str | Path) -> str:
    path = Path(path_like)
    name = path.name
    name = name.replace("_hls", "_src").replace("_py", "_src")
    return name
