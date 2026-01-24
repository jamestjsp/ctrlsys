#!/usr/bin/env python3
"""
Setup script for slicot.c Python bindings.
"""
from setuptools import setup, Extension
import os
import sys
import glob

def get_numpy_include():
    """Defer numpy import until build time."""
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        # Return dummy path, will be populated during install
        return ''

# Platform-aware build directory search
# Only search for libraries matching current platform to avoid cross-platform linking errors
if sys.platform == 'darwin':
    build_dirs = [
        'build/macos-arm64-debug/src',
        'build/macos-arm64-release/src',
        'build/src',
    ]
else:
    # Linux: sanitizer builds first for CI
    build_dirs = [
        'build/linux-x64-debug-sanitizers/src',
        'build/linux-x64-debug-asan/src',
        'build/linux-x64-debug-ubsan/src',
        'build/linux-x64-debug/src',
        'build/linux-x64-release/src',
        'build/src',
    ]

library_dir = None
for bd in build_dirs:
    if os.path.exists(bd):
        libs = glob.glob(os.path.join(bd, 'libslicot.*'))
        if libs:
            library_dir = bd
            break

if library_dir is None:
    raise RuntimeError(
        "Could not find libslicot. Build the C library first:\n"
        "  cmake --preset macos-arm64-debug\n"
        "  cmake --build --preset macos-arm64-debug-build"
    )

print(f"Found slicot library in: {library_dir}")

# Extension modules
wrapper_sources = sorted(glob.glob('python/wrappers/py_*.c'))

ext_modules = [
    Extension(
        'slicot._slicot',
        sources=['python/slicot_module.c'] + wrapper_sources,
        include_dirs=[
            'include',
            'python',
            get_numpy_include()
        ],
        libraries=['slicot'],
        library_dirs=[library_dir],
        runtime_library_dirs=[os.path.abspath(library_dir)],
        extra_compile_args=['-std=c11']
    )
]

setup(
    name='slicot',
    version='0.1.0',
    description='Python bindings for SLICOT C library',
    packages=['slicot'],
    package_dir={'slicot': 'python/slicot'},
    ext_modules=ext_modules,
    setup_requires=[
        'numpy>=1.20.0',
    ],
    install_requires=[
        'numpy>=1.20.0',
    ],
    python_requires='>=3.8',
)
