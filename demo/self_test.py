"""Fast, dependency-light checks for the PLUTO study repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compat"))

from natten import NeighborhoodAttention1D, _window_and_bias_indices  # noqa: E402


def test_natten_indices() -> dict[str, object]:
    keys, bias = _window_and_bias_indices(7, 3, 1, torch.device("cpu"))
    expected_keys = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [4, 5, 6]]
    )
    expected_bias = torch.tensor(
        [[2, 3, 4], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [0, 1, 2]]
    )
    torch.testing.assert_close(keys.cpu(), expected_keys)
    torch.testing.assert_close(bias.cpu(), expected_bias)
    dilated_keys, dilated_bias = _window_and_bias_indices(9, 3, 2, torch.device("cpu"))
    torch.testing.assert_close(
        dilated_keys.cpu(),
        torch.tensor(
            [
                [0, 2, 4],
                [1, 3, 5],
                [0, 2, 4],
                [1, 3, 5],
                [2, 4, 6],
                [3, 5, 7],
                [4, 6, 8],
                [3, 5, 7],
                [4, 6, 8],
            ]
        ),
    )
    torch.testing.assert_close(
        dilated_bias.cpu(),
        torch.tensor(
            [
                [2, 3, 4],
                [2, 3, 4],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [0, 1, 2],
                [0, 1, 2],
            ]
        ),
    )
    return {"name": "natten_indices", "status": "pass", "dilation_2": "pass"}


def test_natten_forward(device: str) -> dict[str, object]:
    torch.manual_seed(7)
    module = NeighborhoodAttention1D(dim=32, num_heads=4, kernel_size=5).to(device)
    x = torch.randn(2, 21, 32, device=device, requires_grad=True)
    y = module(x)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError(f"bad output: shape={tuple(y.shape)}, finite={torch.isfinite(y).all()}")
    y.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("missing or non-finite input gradient")
    short = torch.randn(1, 3, 32, device=device)
    short_output = module(short)
    if short_output.shape != short.shape or not torch.isfinite(short_output).all():
        raise AssertionError("short-sequence padding path failed")
    return {
        "name": "natten_forward_backward",
        "status": "pass",
        "device": device,
        "shape": list(y.shape),
        "short_sequence_padding": "pass",
    }


def test_space_guard(minimum_free_gb: float = 50.0) -> dict[str, object]:
    stat = os.statvfs(ROOT) if hasattr(os, "statvfs") else None
    if stat:
        free_gb = stat.f_bavail * stat.f_frsize / 1024**3
    else:
        import shutil

        free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < minimum_free_gb:
        raise AssertionError(f"free space {free_gb:.2f} GB < guard {minimum_free_gb:.2f} GB")
    return {"name": "space_guard", "status": "pass", "free_gb": round(free_gb, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="do not use CUDA even when available")
    parser.add_argument("--json", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    results = [test_space_guard(), test_natten_indices(), test_natten_forward(device)]
    report = {"status": "pass", "torch": torch.__version__, "results": results}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
