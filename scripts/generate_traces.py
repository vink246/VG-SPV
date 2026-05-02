"""
Thin dispatcher for VG-SPV preferred-response trace generation.

This is a convenience entry point that forwards to the two real generators:

  - ``scripts.generate_method1_traces`` — GPT-5.4-mini (Batch API on
    ``/v1/responses``) Method 1 traces
  - ``scripts.generate_method2_traces`` — Grounding DINO Method 2 traces

Always invoke from the repo root via ``python -m`` so cross-package imports
(``train.*``, ``inference.*``) resolve naturally:

    # Run the full Method 1 pipeline (prepare -> submit -> wait -> collect):
    python -m scripts.generate_traces method1 run --split test

    # Or run a single Method 1 phase:
    python -m scripts.generate_traces method1 prepare --split test
    python -m scripts.generate_traces method1 submit  --split test
    python -m scripts.generate_traces method1 wait    --split test
    python -m scripts.generate_traces method1 collect --split test

    # Run Method 2 (must be after Method 1 collect):
    python -m scripts.generate_traces method2 \\
        --split test --checkpoint weights/groundingdino_swint_ogc.pth

Pass ``--help`` after the subcommand for the full flag list of each generator.
"""

from __future__ import annotations

import runpy
import sys


_USAGE = (
    "Usage: python -m scripts.generate_traces {method1|method2} [args...]\n"
    "  method1  forward to scripts.generate_method1_traces\n"
    "  method2  forward to scripts.generate_method2_traces\n"
    "  -h, --help  show this message"
)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        print()
        print(_USAGE)
        return

    head = argv[0]
    rest = argv[1:]

    if head == "method1":
        target = "scripts.generate_method1_traces"
    elif head == "method2":
        target = "scripts.generate_method2_traces"
    else:
        print(f"Unknown subcommand: {head!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)

    sys.argv = [target] + rest
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
