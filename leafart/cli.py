"""Command-line entry point: ``python -m leafart <stage>``.

Every stage reads from and writes to folders under ``--data`` (default
``data/``), so running the stages in order takes raw scans all the way to
screensaver sprites:

    extract -> triage -> split -> rotate -> web / mosaic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import contact_sheet, extract, mosaic, rotate, split, triage, web


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="leafart", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="data folder (default: data/)")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("extract", help="cut leaves out of data/scans into data/leaves")

    p = sub.add_parser("triage", help="move multi-leaf and cut-off crops out of data/leaves")
    p.add_argument("--dry-run", action="store_true", help="report without moving files")

    sub.add_parser("split", help="separate touching leaves in data/multiple_leaves")

    p = sub.add_parser("rotate", help="turn leaves upright using data/leaf_rotations.csv")
    p.add_argument("--csv", type=Path, help="angles file (default: data/leaf_rotations.csv)")

    p = sub.add_parser("web", help="export screensaver sprites from data/rotated_leaves")
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--max-px", type=int, default=256)

    p = sub.add_parser("mosaic", help="render the radial colour mosaic")
    p.add_argument("--out", type=Path, default=Path("assets/mosaic.jpg"))
    p.add_argument("--size", type=int, default=3000, help="output width/height in pixels")
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("sheet", help="contact sheet of a folder for quality checks")
    p.add_argument("folder", type=Path, nargs="?", help="default: data/leaves")
    p.add_argument("--out", type=Path, default=Path("contact_sheet.jpg"))

    args = parser.parse_args(argv)
    d: Path = args.data

    if args.stage == "extract":
        extract.run(d / "scans", d / "leaves")
    elif args.stage == "triage":
        triage.run(d / "leaves", d / "multiple_leaves", d / "discarded_leaves", dry_run=args.dry_run)
    elif args.stage == "split":
        split.run(d / "multiple_leaves", d / "leaves")
    elif args.stage == "rotate":
        rotate.run(d / "leaves", args.csv or d / "leaf_rotations.csv", d / "rotated_leaves")
    elif args.stage == "web":
        web.run(d / "rotated_leaves", args.out, max_px=args.max_px)
    elif args.stage == "mosaic":
        mosaic.run(d / "rotated_leaves", args.out, mosaic.MosaicSettings(size_px=args.size, seed=args.seed))
    elif args.stage == "sheet":
        contact_sheet.run(args.folder or d / "leaves", args.out)


if __name__ == "__main__":
    main()
