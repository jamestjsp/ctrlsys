#!/usr/bin/env python3
"""Flag known high-risk SLICOT translation patterns."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rule:
    id: str
    path: Path
    pattern: re.Pattern[str]
    message: str
    fix_hint: str


RULES = [
    Rule(
        id="ab13hd-c-row-daxpy-contiguous",
        path=Path("src/AB/ab13hd.c"),
        pattern=re.compile(
            r"SLC_DAXPY\([^;]*&c\[i \* ldc\]\s*,\s*&\(i32\)\{1\}",
            re.DOTALL,
        ),
        message="AB13HD row walk over C appears translated as a contiguous column walk.",
        fix_hint="For Fortran C(I,1), use &c[i] with increment ldc, not &c[i * ldc] with increment 1.",
    ),
    Rule(
        id="mb04hd-bwork-bool-cast",
        path=Path("src/MB/mb04hd.c"),
        pattern=re.compile(r"\(bool \*\)\s*bwork"),
        message="MB04HD casts BWORK to bool*, which can misread an i32 logical workspace bytewise.",
        fix_hint="Make BWORK a bool workspace end-to-end, or pass a real bool selection buffer.",
    ),
]


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_rule(rule: Rule) -> list[str]:
    path = ROOT / rule.path
    text = path.read_text()
    findings = []
    for match in rule.pattern.finditer(text):
        line = line_for_offset(text, match.start())
        findings.append(
            f"{rule.path}:{line}: {rule.id}: {rule.message}\n"
            f"  {rule.fix_hint}"
        )
    return findings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", action="append", choices=[rule.id for rule in RULES])
    args = parser.parse_args(argv)

    selected = [rule for rule in RULES if not args.rule or rule.id in args.rule]
    findings = [finding for rule in selected for finding in check_rule(rule)]
    if findings:
        print("\n".join(findings))
        print(f"\n{len(findings)} translation smell(s) found.")
        return 1

    print(f"No translation smells found across {len(selected)} rule(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
