"""Validate the downloaded PLUTO checkpoint without running a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "_deps" / "pluto-upstream"
sys.path[:0] = [str(ROOT / "compat"), str(UPSTREAM)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=ROOT / "checkpoints" / "pluto_1M_aux_cil.ckpt",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"missing pinned upstream checkout: {UPSTREAM}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    config = OmegaConf.load(UPSTREAM / "config" / "planner" / "pluto_planner.yaml")
    model = instantiate(config.pluto_planner.planner)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = {key.removeprefix("model."): value for key, value in raw["state_dict"].items()}
    incompatible = model.load_state_dict(state, strict=True)

    report = {
        "status": "pass",
        "checkpoint": args.checkpoint.name,
        "sha256": sha256(args.checkpoint),
        "bytes": args.checkpoint.stat().st_size,
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "lightning_version": raw.get("pytorch-lightning_version"),
        "state_tensor_count": len(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "config": {
            "num_heads": config.pluto_planner.planner.num_heads,
            "num_modes": config.pluto_planner.planner.num_modes,
            "history_steps": config.pluto_planner.planner.history_steps,
            "future_steps": config.pluto_planner.planner.future_steps,
            "cat_x": config.pluto_planner.planner.cat_x,
            "ref_free_traj": config.pluto_planner.planner.ref_free_traj,
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
