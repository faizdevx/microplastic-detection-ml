"""`mp` command line: fetch, groups, audit, splits, calibrate."""
from __future__ import annotations

import argparse
import sys

from mp.config import load_config


def _cmd_fetch(args) -> int:
    from mp.data import fetch

    return fetch.run(load_config(args.config), force=args.force)


def _cmd_groups(args) -> int:
    from mp.data import groups

    info = groups.build_groups(load_config(args.config))
    for row in info["sensitivity"]:
        print(row)
    if info["cross_class_pairs"]:
        print(f"WARNING: {len(info['cross_class_pairs'])} cross-class near-duplicate pairs")
    return 0


def _cmd_audit(args) -> int:
    from mp.data import audit

    a = audit.run(load_config(args.config))
    print(a["acquisition_confound_note"])
    return 0


def _cmd_splits(args) -> int:
    from mp.data import splits

    splits.run(load_config(args.config))
    return 0


def _cmd_calibrate(args) -> int:
    from mp.imaging import calibrate_scale as cs

    if args.manual:
        px, um = args.manual
        value = um / px
    else:
        if not args.image or not args.spacing_um:
            print("need IMAGE and --spacing-um, or --manual PX UM", file=sys.stderr)
            return 2
        line = tuple(args.line) if args.line else None
        value = cs.estimate_um_per_px(args.image, args.spacing_um, line)
    print(f"um_per_px = {value:.6f}")
    if not args.dry_run:
        cs.write_um_per_px(value)
        print("written to configs/data.yaml")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mp")
    ap.add_argument("--config", default=None, help="path to data.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download, verify, extract, build manifest")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_fetch)

    sub.add_parser("groups", help="near-duplicate groups + sessions").set_defaults(
        func=_cmd_groups)
    sub.add_parser("audit", help="audit report, figures, shortcut baseline").set_defaults(
        func=_cmd_audit)
    sub.add_parser("splits", help="hold-out, CV and LOSO splits").set_defaults(
        func=_cmd_splits)

    p = sub.add_parser("calibrate", help="measure um/px from a micrometer image")
    p.add_argument("image", nargs="?")
    p.add_argument("--spacing-um", type=float, help="known division spacing in um")
    p.add_argument("--line", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--manual", type=float, nargs=2, metavar=("PX", "UM"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_calibrate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
