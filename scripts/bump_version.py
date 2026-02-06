#!/usr/bin/env python3
"""Bump version in pyproject.toml and meson.build atomically."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
MESON = ROOT / "meson.build"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def read_version(path, pattern):
    text = path.read_text()
    m = re.search(pattern, text)
    if not m:
        sys.exit(f"Could not find version in {path}")
    return m.group(1), text


def main():
    if len(sys.argv) != 2 or not VERSION_RE.match(sys.argv[1]):
        sys.exit("Usage: python scripts/bump_version.py X.Y.Z")

    new = sys.argv[1]

    py_ver, py_text = read_version(PYPROJECT, r'version\s*=\s*"([^"]+)"')
    ms_ver, ms_text = read_version(MESON, r"version:\s*'([^']+)'")

    if py_ver != ms_ver:
        sys.exit(f"Version drift: pyproject.toml={py_ver}, meson.build={ms_ver}")

    if py_ver == new:
        sys.exit(f"Already at {new}")

    PYPROJECT.write_text(py_text.replace(f'version = "{py_ver}"', f'version = "{new}"', 1))
    MESON.write_text(ms_text.replace(f"version: '{ms_ver}'", f"version: '{new}'", 1))

    print(f"{py_ver} -> {new}")


if __name__ == "__main__":
    main()
