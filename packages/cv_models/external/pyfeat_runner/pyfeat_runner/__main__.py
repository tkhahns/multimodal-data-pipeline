from __future__ import annotations

import argparse
import logging
import sys

from .core import run_cli


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Py-Feat detection and emit JSON features.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    return run_cli(args.video)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
