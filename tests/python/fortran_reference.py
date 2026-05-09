"""Helpers for differential tests against the SLICOT Fortran reference."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _candidate_reference_libs() -> list[Path]:
    env_lib = os.environ.get("CTRLSYS_FORTRAN_REFERENCE_LIB")
    candidates = [Path(env_lib)] if env_lib else []
    candidates.extend(
        [
            ROOT / "SLICOT-Reference/build/lib/libslicot.dylib",
            ROOT / "SLICOT-Reference/build/lib/libslicot.so",
            ROOT / "SLICOT-Reference/build/lib/libslicot.a",
            ROOT / "SLICOT-Reference/build_bench/lib/libslicot.dylib",
            ROOT / "SLICOT-Reference/build_bench/lib/libslicot.so",
            ROOT / "SLICOT-Reference/build_bench/lib/libslicot.a",
        ]
    )
    candidates.extend(sorted((ROOT / "SLICOT-Reference/build/lib").glob("libslicot.*.dylib")))
    candidates.extend(sorted((ROOT / "SLICOT-Reference/build_bench/lib").glob("libslicot.*.dylib")))
    return candidates


def reference_library_or_skip() -> Path:
    lib = next((path for path in _candidate_reference_libs() if path.exists()), None)
    if lib is None:
        pytest.skip(
            "SLICOT Fortran reference library is unavailable; build "
            "SLICOT-Reference with CMake or set CTRLSYS_FORTRAN_REFERENCE_LIB"
        )
    return lib


def gfortran_or_skip() -> str:
    compiler = shutil.which(os.environ.get("FC", "gfortran"))
    if compiler is None:
        pytest.skip("gfortran is unavailable; skipping SLICOT Fortran reference comparison")
    return compiler


def run_fortran_driver(source: str, tmp_path: Path) -> str:
    compiler = gfortran_or_skip()
    lib = reference_library_or_skip()
    src = tmp_path / "driver.f90"
    exe = tmp_path / "driver"
    src.write_text(source)

    cmd = [
        compiler,
        "-O0",
        str(src),
        str(lib),
    ]
    extra_flags = os.environ.get("CTRLSYS_FORTRAN_REFERENCE_LINK_FLAGS")
    if extra_flags:
        cmd.extend(shlex.split(extra_flags))
    cmd.extend(["-Wl,-rpath," + str(lib.parent), "-o", str(exe)])

    compiled = subprocess.run(cmd, capture_output=True, text=True)
    if compiled.returncode != 0:
        pytest.fail(
            "failed to compile SLICOT Fortran reference driver:\n"
            + compiled.stdout
            + compiled.stderr
        )

    completed = subprocess.run([str(exe)], capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            "SLICOT Fortran reference driver failed:\n"
            + completed.stdout
            + completed.stderr
        )
    return completed.stdout
