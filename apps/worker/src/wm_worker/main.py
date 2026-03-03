"""Entrypoint for the worker scaffold."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lucid worker scaffold")
    parser.add_argument("--worker-id", default="wm-worker-1")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("Lucid worker scaffold")
    print(f"worker_id={args.worker_id}")
    print("TODO: implement coordinator registration + LiveKit loop from spec.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
