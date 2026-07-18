"""Integrity checks for the compact nuPlan/PLUTO artifact set."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sqlite_quick_check(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        value = connection.execute("PRAGMA quick_check(1)").fetchone()[0]
    finally:
        connection.close()
    if value != "ok":
        raise RuntimeError(f"SQLite quick_check failed for {path}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    db_root = ROOT / "data" / "data" / "cache" / "mini"
    map_root = ROOT / "data" / "maps"
    checkpoint = ROOT / "checkpoints" / "pluto_1M_aux_cil.ckpt"
    databases = sorted(db_root.glob("*.db"))
    maps = sorted(map_root.glob("**/map.gpkg"))

    if len(databases) != 64:
        raise RuntimeError(f"expected 64 mini DB files, got {len(databases)}")
    if len(maps) != 4:
        raise RuntimeError(f"expected 4 map packages, got {len(maps)}")
    if checkpoint.stat().st_size != 51_431_136:
        raise RuntimeError(f"unexpected checkpoint size: {checkpoint.stat().st_size}")

    for path in databases + maps:
        sqlite_quick_check(path)

    report = {
        "status": "pass",
        "nuplan_mini": {
            "database_count": len(databases),
            "bytes": sum(path.stat().st_size for path in databases),
            "sqlite_quick_check": "ok (all files)",
        },
        "maps": {
            "package_count": len(maps),
            "bytes": sum(path.stat().st_size for path in map_root.glob("**/*") if path.is_file()),
            "sqlite_quick_check": "ok (all map.gpkg files)",
        },
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
