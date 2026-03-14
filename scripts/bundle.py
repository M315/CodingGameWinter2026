#!/usr/bin/env python3
"""
Bundle source files into a single CG-submittable Rust file.

Usage:
    python scripts/bundle.py                    # game + beam bot + IO → submission.rs
    python scripts/bundle.py --out foo.rs       # custom output path
    python scripts/bundle.py --bot greedy       # use greedy bot instead of beam
    python scripts/bundle.py --all-bots         # include all bots (for debugging)

The output is a flat single-file Rust program with no module declarations.
"""

import sys
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"

# Lines to strip when inlining files into a flat module
STRIP_PATTERNS = [
    re.compile(r'^\s*use\s+crate::'),
    re.compile(r'^\s*pub\s+use\s+crate::'),
    re.compile(r'^\s*use\s+super::'),
    re.compile(r'^\s*pub\s+use\s+super::'),
    re.compile(r'^\s*pub\s+mod\s+\w+\s*;'),
    re.compile(r'^\s*mod\s+\w+\s*;'),
    re.compile(r'^\s*use\s+snakebyte::'),
    re.compile(r'^\s*use\s+std::'),   # stripped; bundle header provides all std imports
]

def clean(text: str) -> str:
    """Strip crate-relative imports and module declarations."""
    lines = []
    for line in text.splitlines():
        if any(p.match(line) for p in STRIP_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

def bundle(bot: str = "beam", all_bots: bool = False) -> str:
    # Files inlined in order (dependencies first)
    sources = [SRC / "game.rs", SRC / "bots" / "mod.rs"]

    if all_bots:
        sources += [SRC / "bots" / "wait.rs",
                    SRC / "bots" / "greedy.rs",
                    SRC / "bots" / "beam.rs"]
    else:
        # beam always needs greedy_actions (defined in bots/mod.rs already)
        # only include the selected bot's impl file
        if bot == "greedy":
            sources.append(SRC / "bots" / "greedy.rs")
        elif bot == "wait":
            sources.append(SRC / "bots" / "wait.rs")
        elif bot == "old_beam":
            sources.append(SRC / "bots" / "old_beam.rs")
        else:  # beam (default)
            sources.append(SRC / "bots" / "beam.rs")

    sources.append(SRC / "main.rs")

    sections = []
    for path in sources:
        raw = path.read_text(encoding="utf-8")
        sections.append(f"// ── {path.name} " + "─" * (60 - len(path.name)) + "\n" + clean(raw))

    body = "\n\n".join(sections)

    return (
        "#![allow(dead_code, unused_imports, unused_variables)]\n"
        "use std::collections::{HashMap, HashSet, VecDeque};\n"
        "use std::time::{Duration, Instant};\n"
        "use std::io::{self, BufRead};\n\n"
        + body
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle Rust source for CG submission")
    parser.add_argument("--out",      default=str(ROOT / "submission.rs"), help="Output path")
    parser.add_argument("--bot",      default="beam", choices=["wait", "greedy", "beam", "old_beam"])
    parser.add_argument("--all-bots", action="store_true", help="Include all bot implementations")
    args = parser.parse_args()

    result = bundle(bot=args.bot, all_bots=args.all_bots)

    out = Path(args.out)
    out.write_text(result, encoding="utf-8")
    lines = result.count("\n")
    print(f"Written {lines} lines → {out}", file=sys.stderr)
