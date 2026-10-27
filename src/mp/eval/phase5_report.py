"""results/phase5_summary.md and the Phase 5 section of MODEL_CARD.md (generated)."""
from __future__ import annotations

import json
import re

from mp.config import repo_root, resolve

ORDER = ["baseline", "V1_local_normalisation", "V2_exposure_randomisation", "V3_patch_cnn"]


def verdict(res: dict, rcfg: dict) -> dict:
    gain, drop = rcfg["verdict"]["cross_res_min_gain"], rcfg["verdict"]["pseudo_min_drop"]

    def xres(r):
        c = r["cross_resolution"]
        return (c["a_to_b"]["balanced_accuracy"] + c["b_to_a"]["balanced_accuracy"]) / 2

    base = res["baseline"]
    out = {}
    for n in ORDER[1:]:
        if n not in res:
            continue
        r = res[n]
        dx = xres(r) - xres(base)
        dp = r["pseudo_particles"]["balanced_accuracy"] - base["pseudo_particles"]["balanced_accuracy"]
        x_v = "improves" if dx >= gain else ("worse" if dx <= -gain else "no change")
        p_v = "improves" if dp <= -drop else ("worse" if dp >= drop else "no change")
        out[n] = {"cross_resolution": x_v, "cross_resolution_delta": dx,
                  "background_control": p_v, "background_control_delta": dp,
                  "overall": "improves" if (x_v == "improves" and p_v == "improves")
                  else ("worse" if "worse" in (x_v, p_v) else "no clear improvement")}
    return out


def _fmt(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def write(variants_json: dict, probe: dict) -> tuple[str, dict]:
    res, rcfg = variants_json["results"], variants_json["config"]
    ver = verdict(res, rcfg)
    p2 = json.loads((resolve("results") / "phase2_controls.json").read_text(encoding="utf8"))
    bg_img = p2["geometries"]["resample"]["background_only"]["balanced_accuracy"]
    L = ["## Phase 5: robustness of polymer identification (generated)", "",
         "Cross-validation pool only; the hold-out was not read. `resample` geometry, "
         f"{variants_json['n_cv_images']} images.", "",
         "| Variant | CV accuracy | 640→1280 bal. acc. | 1280→640 bal. acc. | Res.-matched bal. acc. "
         "| Background discs bal. acc. | Permutation p |",
         "|---|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in res:
            continue
        r = res[n]
        L.append(f"| {n} | {_fmt(r['cv']['accuracy'])} | "
                 f"{_fmt(r['cross_resolution']['a_to_b']['balanced_accuracy'])} | "
                 f"{_fmt(r['cross_resolution']['b_to_a']['balanced_accuracy'])} | "
                 f"{_fmt(r['resolution_matched']['balanced_accuracy'])} | "
                 f"{_fmt(r['pseudo_particles']['balanced_accuracy'])} | "
                 f"{_fmt(r['permutation']['p_value'])} |")
    L += ["", "Background discs: discs of class-independent size placed on pixels that touch no particle; "
          "the same pipeline is trained and tested on them. Chance is 0.5. "
          f"The Phase 2 image-level background control gave {_fmt(bg_img)}.", "",
          "### Acquisition probes (higher than chance = the representation carries the cue)", "",
          "| Variant | Exposure, overall | Exposure, within class | Resolution "
          "| Mean abs. Spearman ρ, P(NYLON) vs brightness |",
          "|---|---|---|---|---|"]
    for n in ORDER:
        if n not in probe:
            continue
        p = probe[n]["probe"]
        L.append(f"| {n} | {_fmt(p['exposure_overall']['accuracy'])} (chance "
                 f"{_fmt(p['exposure_overall']['chance'])}) | "
                 f"{_fmt(p['exposure_within_class']['accuracy'])} (chance "
                 f"{_fmt(p['exposure_within_class']['chance'])}) | "
                 f"{_fmt(p['resolution']['accuracy'])} (chance {_fmt(p['resolution']['chance'])}) | "
                 f"{_fmt(probe[n]['prediction_coupling']['mean_abs'])} |")
    L += ["", "### Verdict against the baseline", "",
          f"Thresholds: cross-resolution balanced accuracy must rise by at least "
          f"{rcfg['verdict']['cross_res_min_gain']}, background-disc balanced accuracy must fall by at "
          f"least {rcfg['verdict']['pseudo_min_drop']}.", "",
          "| Variant | Cross-resolution | Background discs | Overall |", "|---|---|---|---|"]
    for n, v in ver.items():
        L.append(f"| {n} | {v['cross_resolution']} ({v['cross_resolution_delta']:+.3f}) | "
                 f"{v['background_control']} ({v['background_control_delta']:+.3f}) | {v['overall']} |")
    better = [n for n, v in ver.items() if v["overall"] == "improves"]
    sets_p = resolve("results") / "phase5_feature_sets.json"
    if sets_p.exists():
        sets = json.loads(sets_p.read_text(encoding="utf8"))["results"]
        L += ["", "### Which features carry the signal (same model, one feature set at a time)", "",
              "| Feature set | CV accuracy | 640→1280 bal. acc. | 1280→640 bal. acc. "
              "| Background discs bal. acc. |",
              "|---|---|---|---|---|"]
        for n, v in sets.items():
            L.append(f"| {n} | {_fmt(v['cv']['accuracy'])} | "
                     f"{_fmt(v['cross_resolution']['a_to_b']['balanced_accuracy'])} | "
                     f"{_fmt(v['cross_resolution']['b_to_a']['balanced_accuracy'])} | "
                     f"{_fmt(v['background_discs']['balanced_accuracy'])} |")
        sh = sets["shape"]
        pix = {k: v for k, v in sets.items() if k != "shape"}
        lo = min(v["background_discs"]["balanced_accuracy"] for v in pix.values())
        hi = max(v["background_discs"]["balanced_accuracy"] for v in pix.values())
        L += ["", f"Shape features alone give CV accuracy {_fmt(sh['cv']['accuracy'])} and cross-resolution "
              f"balanced accuracy {_fmt(sh['cross_resolution']['a_to_b']['balanced_accuracy'])} / "
              f"{_fmt(sh['cross_resolution']['b_to_a']['balanced_accuracy'])}; on background "
              f"discs they sit at {_fmt(sh['background_discs']['balanced_accuracy'])}, as they should "
              "because the discs have no "
              "class-dependent shape. Every feature set built from pixel values separates the classes on "
              f"background discs too ({_fmt(lo)} to {_fmt(hi)}), so their accuracy mixes particle and "
              "background information."]
    L += ["", ("Variants that improve on both controls: " + ", ".join(better) + "." if better else
               "No variant improves on both controls; the Phase 2 model stays the default in `mp analyse`."),
          ""]
    text = "\n".join(L) + "\n"
    (resolve("results") / "phase5_summary.md").write_text(text, encoding="utf8")
    return text, ver


def update_model_card(summary_text: str) -> None:
    p = repo_root() / "MODEL_CARD.md"
    card = p.read_text(encoding="utf8")
    body = summary_text.replace("## Phase 5: robustness of polymer identification (generated)",
                                "## Phase 5: robustness analysis", 1)
    body += ("\nAnalyses that used the cross-validation pool only: all Phase 5 variants, controls, "
             "probes and the verdict. The hold-out (`results/phase2_holdout.json`) was evaluated once "
             "in Phase 2 and not touched again.\n")
    if "## Phase 5: robustness analysis" in card:
        card = re.sub(r"## Phase 5: robustness analysis.*\Z", lambda m: body, card, flags=re.S)
    else:
        card = card.rstrip() + "\n\n" + body
    p.write_text(card, encoding="utf8", newline="\n")
