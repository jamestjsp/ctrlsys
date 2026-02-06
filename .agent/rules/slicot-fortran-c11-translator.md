---
trigger: always_on
---

C11 translation of SLICOT (Subroutine Library In Control Theory) from Fortran77.

**Source of truth:** See `/CLAUDE.md` for complete instructions.

**Reference:** `SLICOT-Reference/src/` (Fortran77), `SLICOT-Reference/doc/` (HTML docs)

## Quick Commands (macOS)

```bash
# Setup
uv venv && uv pip install ".[test]"

# Build & test
cmake --preset macos-arm64-debug
cmake --build --preset macos-arm64-debug-build
uv pip install .
.venv/bin/pytest tests/python/ -v

# Clean rebuild
rm -rf build/macos-arm64-debug && cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug-build && uv pip install .
```

## Memory Debugging

```bash
# ASAN via Docker (required before PR)
./scripts/run_asan_docker.sh                           # full suite
./scripts/run_asan_docker.sh tests/python/test_x.py -v # single file

# macOS quick checks (assumes venv active)
MallocScribble=1 pytest tests/python/ -v               # use-after-free
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib pytest tests/python/test_x.py -v  # overflow
```

## Translation Workflow

> [!NOTE]
> See `/.agent/workflows/translate_slicot_routine.md` for the step-by-step TDD process.

**Check deps:** `python3 tools/extract_dependencies.py SLICOT-Reference/src/ ROUTINE_NAME`
