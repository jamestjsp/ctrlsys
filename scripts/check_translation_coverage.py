#!/usr/bin/env python3
"""Run targeted C line coverage for high-risk translated SLICOT routines."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TESTS = [
    "tests/python/test_ab13hd.py",
    "tests/python/test_mb04hd.py",
]

DEFAULT_TARGETS = [
    ("src/AB/ab13hd.c", 1842, "AB13HD C-row access branch"),
    ("src/MB/mb04hd.c", 228, "MB04HD reorder/BWORK call"),
]


@dataclass(frozen=True)
class Target:
    source: Path
    line: int
    label: str


@dataclass(frozen=True)
class LineCoverage:
    target: Target
    count: int | None
    text: str

    @property
    def covered(self) -> bool:
        return self.count is not None and self.count > 0


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def parse_target(value: str) -> Target:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("target must be SOURCE:LINE[:LABEL]")

    source = Path(parts[0])
    try:
        line = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target line must be an integer") from exc

    label = parts[2] if len(parts) == 3 else f"{source}:{line}"
    return Target(source=source, line=line, label=label)


def default_targets() -> list[Target]:
    return [Target(Path(source), line, label) for source, line, label in DEFAULT_TARGETS]


def clean_build_dirs() -> None:
    build = ROOT / "build"
    if build.exists():
        for path in build.iterdir():
            if path.is_dir() and re.fullmatch(r"cp\d+.*", path.name):
                shutil.rmtree(path)


def coverage_env(venv: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("CC", "clang")
    env["CFLAGS"] = "-O0 -g --coverage"
    env["LDFLAGS"] = "--coverage"
    return env


def build_extension(venv: Path, clean: bool) -> None:
    if clean:
        shutil.rmtree(venv, ignore_errors=True)
        clean_build_dirs()

    run(["uv", "venv", str(venv)])
    py = venv / "bin" / "python"
    run([
        "uv",
        "pip",
        "install",
        "--python",
        str(py),
        "pip",
        "numpy>=2.0",
        "meson-python",
        "meson>=1.1.0",
        "ninja",
        "pytest>=7.0.0",
        "pytest-xdist>=3.0.0",
        "pytest-rerunfailures>=14.0",
    ])
    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-e",
        ".",
    ], env=coverage_env(venv))


def get_build_dir(venv: Path) -> Path:
    py = venv / "bin" / "python"
    code = "import ctrlsys._slicot as m; from pathlib import Path; print(Path(m.__file__).parents[1])"
    out = subprocess.check_output(
        [str(py), "-c", code],
        cwd=ROOT,
        env=coverage_env(venv),
        text=True,
    ).strip()
    return Path(out)


def clear_coverage_outputs(build_dir: Path) -> None:
    for pattern in ("*.gcda", "*.gcov"):
        for path in build_dir.rglob(pattern):
            path.unlink()


def run_tests(venv: Path, tests: list[str], build_dir: Path) -> None:
    clear_coverage_outputs(build_dir)
    run([
        str(venv / "bin" / "python"),
        "-m",
        "pytest",
        *tests,
        "-q",
        "--override-ini",
        "addopts=",
    ], env=coverage_env(venv))


def gcno_for(build_dir: Path, source: Path) -> Path:
    source = source.as_posix()
    match = re.fullmatch(r"src/([^/]+)/([^/]+)\.c", source)
    if not match:
        raise ValueError(f"cannot map target source to Meson object path: {source}")
    family, routine = match.groups()
    return build_dir / "src" / "libslicot.a.p" / f"{family}_{routine}.c.gcno"


def generate_gcov(build_dir: Path, targets: list[Target]) -> None:
    for target in targets:
        gcno = gcno_for(build_dir, target.source)
        if not gcno.exists():
            raise FileNotFoundError(gcno)
        run(["xcrun", "llvm-cov", "gcov", str(gcno.relative_to(build_dir))], cwd=build_dir)


def parse_gcov_line(build_dir: Path, target: Target) -> LineCoverage:
    gcov_path = build_dir / f"{target.source.name}.gcov"
    if not gcov_path.exists():
        raise FileNotFoundError(gcov_path)

    for raw_line in gcov_path.read_text().splitlines():
        fields = raw_line.split(":", 2)
        if len(fields) != 3:
            continue
        if fields[1].strip() != str(target.line):
            continue

        marker = fields[0].strip()
        text = fields[2].strip()
        if marker in {"-", "#####", "====="}:
            return LineCoverage(target, 0 if marker != "-" else None, text)
        return LineCoverage(target, int(marker.rstrip("*")), text)

    return LineCoverage(target, None, "<line not present in gcov output>")


def report(results: list[LineCoverage]) -> int:
    width = max(len(str(item.target.source)) for item in results)
    failures = 0
    for item in results:
        location = f"{item.target.source}:{item.target.line}".ljust(width + 8)
        status = "covered" if item.covered else "UNCOVERED"
        count = "-" if item.count is None else str(item.count)
        print(f"{status:9} count={count:>5} {location} {item.target.label}")
        print(f"          {item.text}")
        if not item.covered:
            failures += 1
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=".venv-gcov", type=Path)
    parser.add_argument("--no-build", action="store_true", help="reuse the existing coverage editable build")
    parser.add_argument("--no-clean", action="store_true", help="do not remove the coverage venv/build before building")
    parser.add_argument("--report-only", action="store_true", help="print uncovered lines but exit 0")
    parser.add_argument("--target", action="append", type=parse_target, help="SOURCE:LINE[:LABEL]")
    parser.add_argument("tests", nargs="*", help="pytest files or node ids; defaults to AB13HD and MB04HD tests")
    args = parser.parse_args(argv)

    targets = args.target or default_targets()
    tests = args.tests or DEFAULT_TESTS
    venv = (ROOT / args.venv).resolve()

    if not args.no_build:
        build_extension(venv, clean=not args.no_clean)

    build_dir = get_build_dir(venv)
    run_tests(venv, tests, build_dir)
    generate_gcov(build_dir, targets)
    failures = report([parse_gcov_line(build_dir, target) for target in targets])
    if failures:
        print(f"\n{failures} target line(s) were not executed.")
    return 0 if args.report_only else min(failures, 1)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
