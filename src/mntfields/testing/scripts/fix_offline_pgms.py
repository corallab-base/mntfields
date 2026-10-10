#!/usr/bin/env python3
import argparse
from pathlib import Path


def _is_broken_header(data: bytes) -> bool:
    # Broken files were written with escaped newlines in header: b"P5\\n..."
    return data.startswith(b"P5\\n")


def _fix_header_bytes(data: bytes) -> bytes:
    # Replace only the first three escaped newlines used by the PGM header.
    fixed = data
    for _ in range(3):
        fixed = fixed.replace(b"\\n", b"\n", 1)
    return fixed


def fix_pgm_file(path: Path) -> str:
    raw = path.read_bytes()
    if _is_broken_header(raw):
        fixed = _fix_header_bytes(raw)
        path.write_bytes(fixed)
        return "fixed"
    if raw.startswith(b"P5\n"):
        return "ok"
    return "unknown"


def find_map_pgms(root: Path):
    for pgm in sorted(root.glob("session_*/ros_map/map.pgm")):
        if pgm.is_file():
            yield pgm


def main():
    parser = argparse.ArgumentParser(
        description="Fix broken offline map.pgm headers under offline_datasets."
    )
    parser.add_argument(
        "--root",
        default="/home/mayank/turtlebot3_ws/mntfields_model/offline_datasets",
        help="Root folder containing session_* directories.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    fixed = 0
    ok = 0
    unknown = 0
    files = list(find_map_pgms(root))
    if not files:
        print(f"No map.pgm files found under {root}")
        return

    for pgm in files:
        status = fix_pgm_file(pgm)
        if status == "fixed":
            fixed += 1
        elif status == "ok":
            ok += 1
        else:
            unknown += 1
        print(f"[{status.upper():7}] {pgm}")

    print(f"\nSummary: total={len(files)} fixed={fixed} ok={ok} unknown={unknown}")


if __name__ == "__main__":
    main()
