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


def _cmd_features(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp.config import load_named
    from mp.imaging import features

    cfg, ccfg = load_config(args.config), load_named("classifier")
    for g in args.geometry or ccfg["geometries"]:
        p, b = features.build_tables(g, cfg, ccfg, n_jobs=args.jobs)
        print(f"{g}: {len(p)} particles in {p['rel_path'].nunique()} images")
    return 0


def _cmd_segment(args) -> int:
    """Auto masks (YOLO-seg labels + resampled images) for every unique image."""
    import warnings

    warnings.filterwarnings("ignore")
    from mp.annotate import auto

    df = auto.export_auto(load_config(args.config), limit=args.limit)
    print(f"auto labels for {len(df)} images; mean {df['n_auto_particles'].mean():.1f} particles")
    return 0


def _cmd_train_cls(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp import pipeline
    from mp.eval import explain

    cfg = load_config(args.config)
    pipeline.train_cls(cfg)
    explain.explain(cfg)
    return 0


def _cmd_controls(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp import pipeline

    pipeline.controls(load_config(args.config), n_jobs=args.jobs)
    return 0


def _cmd_annotate_export(args) -> int:
    from mp.annotate import auto

    print(auto.export_correction_set(args.n_per_class, cfg=load_config(args.config)))
    return 0


def _cmd_annotate_import(args) -> int:
    from mp.annotate import auto

    print(auto.import_corrected(args.coco, load_config(args.config)))
    return 0


def _cmd_train_seg(args) -> int:
    from mp.models import yolo

    cfg = load_config(args.config)
    if args.smoke:
        r = yolo.train_and_eval("smoke", smoke=True, cfg=cfg)
        print("smoke ok:", r["seg"], r["labels"]["description"])
        return 0
    keys = ["all_cv"] if args.all else [args.data or f"fold_{args.fold}"]
    for k in keys:
        r = yolo.train_and_eval(k, epochs=args.epochs, cfg=cfg)
        print(k, r["seg"], r["labels"]["description"])
    return 0


def _cmd_report(args) -> int:
    from mp.eval import report

    cfg = load_config(args.config)
    if args.holdout:
        try:
            report.run_holdout(i_know=args.i_know, cfg=cfg)
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1
    print(report.write_summary(cfg))
    return 0


def _cmd_scatter_generate(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp.scatter import generate

    generate.run()
    return 0


def _cmd_scatter_analyse(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp.scatter import analysis

    r = analysis.run()
    print("analysis written; most sensitive:", r["geometry_sensitivity"]["most_sensitive"])
    return 0


def _cmd_train_reg(args) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from mp.models import regression

    exp = {"E1": "E1", "robustness": "robustness", "all": "all"}[args.experiment]
    regression.run(exp)
    return 0


def _cmd_report3(args) -> int:
    from mp.scatter import report

    print(report.write_summary())
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

    p = sub.add_parser("features", help="per-particle + background features")
    p.add_argument("--geometry", nargs="*", choices=["resample", "native"])
    p.add_argument("--jobs", type=int, default=-1)
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("segment", help="auto masks -> YOLO-seg labels + resampled images")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=_cmd_segment)

    sub.add_parser("train-cls", help="tuned CV, model selection, explainability").set_defaults(
        func=_cmd_train_cls)

    p = sub.add_parser("controls", help="confound controls for the selected model")
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=_cmd_controls)

    p = sub.add_parser("annotate-export", help="COCO set for human correction")
    p.add_argument("--n-per-class", type=int, default=150)
    p.set_defaults(func=_cmd_annotate_export)

    p = sub.add_parser("annotate-import", help="corrected COCO -> labels + IoU")
    p.add_argument("coco")
    p.set_defaults(func=_cmd_annotate_import)

    p = sub.add_parser("train-seg", help="YOLOv11-seg training on splits.json folds")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all", action="store_true", help="train on all CV images")
    p.add_argument("--data", help="dataset key, e.g. res_matched_0, cross_a_to_b")
    p.add_argument("--epochs", type=int)
    p.add_argument("--smoke", action="store_true", help="2 epochs, 20 images, CPU")
    p.set_defaults(func=_cmd_train_seg)

    p = sub.add_parser("report-phase2", help="summary markdown; --holdout evaluates once")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--i-know", action="store_true", help="allow re-evaluating the hold-out")
    p.set_defaults(func=_cmd_report)

    sub.add_parser("scatter-generate", help="multi-angle scattering table").set_defaults(
        func=_cmd_scatter_generate)
    sub.add_parser("scatter-analyse", help="angular curves, contrast, linearity, limits"
                   ).set_defaults(func=_cmd_scatter_analyse)
    p = sub.add_parser("train-reg", help="concentration regression")
    p.add_argument("--experiment", choices=["E1", "robustness", "all"], default="all")
    p.set_defaults(func=_cmd_train_reg)
    sub.add_parser("report-phase3", help="summary markdown").set_defaults(func=_cmd_report3)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
