#!/usr/bin/env python3
"""
Extract usage examples from test files for docstrings.

Usage:
    python tools/extract_examples.py                    # Extract all
    python tools/extract_examples.py --prefix ab        # Only AB* routines
    python tools/extract_examples.py --dry-run          # Show without writing
"""

import ast
import json
import argparse
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests" / "python"
JSON_FILE = PROJECT_ROOT / "python" / "data" / "docstrings.json"

PRIORITY_TEST_NAMES = ["test_html_doc_example", "test_basic", "test_example"]


def extract_routine_name(filepath: Path) -> Optional[str]:
    """Extract routine name from test filename (test_ab01md.py -> ab01md)."""
    name = filepath.stem
    if name.startswith("test_"):
        return name[5:]
    return None


def find_best_test_function(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """Find the best test function to use as example."""
    tests = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if not tests:
        return None

    for priority_name in PRIORITY_TEST_NAMES:
        for test in tests:
            if test.name == priority_name:
                return test

    return tests[0]


def extract_example_lines(func: ast.FunctionDef, source_lines: list[str]) -> list[str]:
    """Extract relevant code lines from a test function."""
    start_line = func.lineno - 1
    end_line = func.end_lineno

    func_lines = source_lines[start_line:end_line]

    body_lines = []
    in_body = False
    in_docstring = False
    docstring_delim = None
    base_indent = None
    skip_until_closed = 0

    for line in func_lines:
        stripped = line.lstrip()
        if not in_body:
            if stripped.startswith('def '):
                in_body = True
            continue

        if not stripped or stripped.startswith('#'):
            continue

        if in_docstring:
            if docstring_delim and docstring_delim in stripped:
                in_docstring = False
                docstring_delim = None
            continue

        if stripped.startswith('"""'):
            if stripped.count('"""') == 1:
                in_docstring = True
                docstring_delim = '"""'
            continue
        if stripped.startswith("'''"):
            if stripped.count("'''") == 1:
                in_docstring = True
                docstring_delim = "'''"
            continue

        if base_indent is None:
            base_indent = len(line) - len(stripped)

        if skip_until_closed > 0:
            skip_until_closed += stripped.count('[') + stripped.count('(')
            skip_until_closed -= stripped.count(']') + stripped.count(')')
            continue

        if stripped.startswith('assert '):
            continue
        if stripped.startswith('np.testing.assert'):
            continue
        if stripped.startswith('pytest.'):
            continue
        if 'assert_allclose' in stripped:
            continue
        if 'assert_array' in stripped:
            continue
        if stripped.startswith('for ') and stripped.endswith(':'):
            continue
        if stripped.startswith('if ') and stripped.endswith(':'):
            continue
        if stripped.startswith('while ') and stripped.endswith(':'):
            continue
        if stripped.startswith('with ') and stripped.endswith(':'):
            continue
        if '_expected' in stripped or '_copy' in stripped or '_orig' in stripped:
            open_count = stripped.count('[') + stripped.count('(')
            close_count = stripped.count(']') + stripped.count(')')
            if open_count > close_count:
                skip_until_closed = open_count - close_count
            continue

        current_indent = len(line) - len(stripped)
        relative_indent = max(0, current_indent - base_indent)

        body_lines.append(' ' * relative_indent + stripped)

    return body_lines


def simplify_example(lines: list[str], routine_name: str) -> list[str]:
    """Simplify example by removing unnecessary setup."""
    result = []
    seen_import_numpy = False
    seen_import_slicot = False
    info_var = None

    for line in lines:
        if 'import numpy' in line or 'from numpy' in line:
            if not seen_import_numpy:
                result.append("import numpy as np")
                seen_import_numpy = True
            continue

        if 'from ctrlsys import' in line or 'import ctrlsys' in line:
            if not seen_import_slicot:
                result.append(f"from ctrlsys import {routine_name}")
                seen_import_slicot = True
            continue

        if 'np.random.seed' in line:
            continue

        if f'{routine_name}(' in line and '=' in line:
            lhs = line.split('=')[0].strip()
            parts = [p.strip() for p in lhs.split(',')]
            for p in parts:
                if p == 'info':
                    info_var = 'info'
                    break

        result.append(line)

    if not seen_import_numpy:
        result.insert(0, "import numpy as np")
    if not seen_import_slicot:
        idx = 1 if seen_import_numpy else 0
        result.insert(idx, f"from ctrlsys import {routine_name}")

    if info_var:
        result.append(info_var)
        result.append("0")

    return result


def format_doctest(lines: list[str]) -> str:
    """Format lines as doctest with >>> prefix."""
    result = []
    in_multiline = False
    open_brackets = 0
    prev_was_bare_name = False

    for line in lines:
        if not line.strip():
            result.append("")
            prev_was_bare_name = False
            continue

        is_output = prev_was_bare_name and line.strip().replace('.', '').replace('-', '').isdigit()
        prev_was_bare_name = False

        if is_output:
            result.append(line)
            continue

        if in_multiline or line.startswith(' '):
            result.append(f"...     {line.lstrip()}")
        else:
            result.append(f">>> {line}")

        if line.strip().isidentifier():
            prev_was_bare_name = True

        open_brackets += line.count('[') + line.count('(') + line.count('{')
        open_brackets -= line.count(']') + line.count(')') + line.count('}')
        in_multiline = open_brackets > 0

    return "\n".join(result)


def extract_example_from_file(filepath: Path) -> Optional[str]:
    """Extract example from a test file."""
    routine_name = extract_routine_name(filepath)
    if not routine_name:
        return None

    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except Exception:
        return None

    test_func = find_best_test_function(tree)
    if not test_func:
        return None

    source_lines = source.splitlines()
    example_lines = extract_example_lines(test_func, source_lines)

    if not example_lines:
        return None

    simplified = simplify_example(example_lines, routine_name)

    if len(simplified) < 3:
        return None

    has_call = any(f'{routine_name}(' in line for line in simplified)
    if not has_call:
        return None

    return format_doctest(simplified)


def main():
    parser = argparse.ArgumentParser(description='Extract examples from tests')
    parser.add_argument('--prefix', type=str, help='Only process routines with this prefix')
    parser.add_argument('--dry-run', action='store_true', help='Show without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--force', action='store_true', help='Replace existing examples')
    args = parser.parse_args()

    if not JSON_FILE.exists():
        print(f"ERROR: {JSON_FILE} not found")
        return 1

    with open(JSON_FILE) as f:
        docstrings = json.load(f)

    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    if args.prefix:
        test_files = [f for f in test_files if f.stem.startswith(f"test_{args.prefix}")]

    extracted = 0
    skipped = 0

    for filepath in test_files:
        routine_name = extract_routine_name(filepath)
        if not routine_name:
            continue

        if routine_name not in docstrings:
            if args.verbose:
                print(f"SKIP {routine_name}: not in docstrings.json")
            skipped += 1
            continue

        example = extract_example_from_file(filepath)
        if not example:
            if args.verbose:
                print(f"SKIP {routine_name}: no suitable example found")
            skipped += 1
            continue

        current_doc = docstrings[routine_name]
        if isinstance(current_doc, dict):
            current_doc = current_doc.get("docstring", "")

        has_example = "Example:" in current_doc or "Examples:" in current_doc
        if has_example and not args.force:
            if args.verbose:
                print(f"SKIP {routine_name}: already has example")
            skipped += 1
            continue

        if has_example:
            base_doc = current_doc.split("\nExample:")[0].split("\nExamples:")[0]
        else:
            base_doc = current_doc

        new_doc = base_doc.rstrip() + "\n\nExample:\n" + example
        docstrings[routine_name] = new_doc

        extracted += 1
        if args.verbose:
            print(f"OK   {routine_name}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would update {extracted} docstrings (skipped {skipped})")
        if extracted > 0 and args.verbose:
            sample = list(docstrings.keys())[0]
            print(f"\nSample ({sample}):\n{docstrings[sample][:500]}...")
    else:
        with open(JSON_FILE, 'w') as f:
            json.dump(docstrings, f, indent=2)
        print(f"Updated {extracted} docstrings in {JSON_FILE.name} (skipped {skipped})")

    return 0


if __name__ == '__main__':
    exit(main())
