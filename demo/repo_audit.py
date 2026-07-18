"""Audit the publishable tutorial surface without requiring private artifacts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def publishable_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode == 0 and result.stdout.strip():
        return [ROOT / line for line in result.stdout.splitlines()]
    ignored = {"_deps", "data", "downloads", "checkpoints", ".git"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts
        if any(part in ignored for part in parts):
            continue
        # Before git init, mirror .gitignore: retain only reviewed, compact
        # files directly under demo/outputs and exclude raw nested runs.
        if len(parts) >= 2 and parts[:2] == ("demo", "outputs") and len(parts) > 3:
            continue
        files.append(path)
    return files


def check_markdown_links(files: list[Path]) -> dict[str, object]:
    checked = 0
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"broken link in {path.relative_to(ROOT)}: {target}")
    return {"name": "markdown_local_links", "status": "pass", "checked": checked}


def check_publish_scope(files: list[Path]) -> dict[str, object]:
    forbidden = {"_deps", "data", "downloads", "checkpoints"}
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in forbidden:
            raise RuntimeError(f"private/third-party artifact would be published: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise RuntimeError(f"unexpected publishable file >10 MiB: {relative}")
    return {"name": "publish_scope", "status": "pass", "file_count": len(files)}


def check_json() -> dict[str, object]:
    paths = [ROOT / "docs" / "artifact_manifest.json", *sorted((ROOT / "demo" / "outputs").glob("*.json"))]
    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))
    return {"name": "json_parse", "status": "pass", "file_count": len(paths)}


def check_python(files: list[Path]) -> dict[str, object]:
    paths = [path for path in files if path.suffix == ".py"]
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return {"name": "python_syntax", "status": "pass", "file_count": len(paths)}


def main() -> int:
    files = publishable_files()
    report = {
        "status": "pass",
        "results": [
            check_markdown_links(files),
            check_publish_scope(files),
            check_json(),
            check_python(files),
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
