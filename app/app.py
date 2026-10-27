"""Local demo: polymer from a fluorescence image, concentration from four detector counts.

    python app/app.py [--port 7860] [--share false]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mp import analyse  # noqa: E402
from mp.config import repo_root  # noqa: E402

ROOT = repo_root()
EXAMPLES = Path(__file__).parent / "examples"
NOTE = analyse.ESTIMATE_NOTE


def polymer_tab(image_path):
    if not image_path:
        return "Upload a fluorescence image.", None, None
    r = analyse.analyse_image(image_path, with_overlay=True)
    geo, labels = r.pop("_overlay")
    vis = geo.copy()
    for i in range(1, int(labels.max()) + 1):
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    p, pc = r["polymer"], r["particles"]
    if p["label"]:
        head = (f"**{p['label']}**: P(nylon) {p['p_nylon']:.2f}, confidence {p['confidence']:.2f}\n\n"
                f"{p['confidence_note']}.")
    else:
        head = f"**Cannot decide.** {p.get('reason', '')}"
    text = (head + f"\n\nParticles found: {pc['count']}. "
            f"{pc['pixel_size_note'] or ''}\n\n" + "\n".join(f"- {w}" for w in r["warnings"]))
    h = pc["size_hist_px"]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    labels_x = [f"{int(a)}-{int(b)}" for a, b in zip(h["bin_edges_px"][:-2], h["bin_edges_px"][1:-1])]
    labels_x.append(f">{int(h['bin_edges_px'][-2])}")
    ax.bar(labels_x, h["counts"], color="#4c72b0")
    ax.set_xlabel("equivalent diameter (px)")
    ax.set_ylabel("particles")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return text, vis, fig


def concentration_tab(s1, s2, s3, s4, adc):
    try:
        c = analyse.estimate_concentration([s1, s2, s3, s4], adc)
    except FileNotFoundError as e:
        return f"{e}", None
    if c["mg_L"] is None:
        return f"**No estimate.** {c['reason']}\n\n{NOTE}", None
    lo, hi = c["interval_90"]
    text = (f"**{c['mg_L']:.2f} mg/L** (90 % interval {lo:.2f} to {hi:.2f})\n\n"
            f"Model: {c['model']}, geometry {c['geometry']}, ADC {c['adc']}.\n\n{NOTE}.")
    curve = analyse.calibration_curve(adc)
    conc = np.array(curve["conc_mg_L"])
    sig = np.array(curve["counts_above_blank"])
    blank = np.array(curve["blank_counts"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, t in enumerate(("S1", "S2")):
        ax.loglog(conc, np.maximum(sig[:, i], 1e-2), label=f"{t} (rig model)")
    ds = np.array([s1, s2]) - blank[:2]
    ax.plot([c["mg_L"]] * 2, np.maximum(ds, 1e-2), "ko", label="your input")
    ax.axvspan(lo, hi, color="gray", alpha=0.2, label="90 % interval")
    ax.set_xlabel("concentration (mg/L)")
    ax.set_ylabel("counts above blank")
    ax.set_title("Rig model calibration curve (default geometry)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return text, fig


def _sections(md: str, names: list[str]) -> str:
    out = []
    for n in names:
        m = re.search(rf"^## {re.escape(n)}\n(.*?)(?=^## |\Z)", md, flags=re.S | re.M)
        if m:
            out.append(f"### {n}\n{m.group(1).strip()}")
    return "\n\n".join(out)


def about_text() -> str:
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf8")
    parts = [_sections(card, ["Not for", "Failure modes", "Not evaluated"])]
    ctl_p = ROOT / "results" / "phase2_controls.json"
    if ctl_p.exists():
        t = json.loads(ctl_p.read_text(encoding="utf8"))["geometries"]["resample"]
        rm, a, b = (t["resolution_matched"], t["cross_resolution"]["a_to_b"],
                    t["cross_resolution"]["b_to_a"])
        parts.append(
            "### Confound controls (balanced accuracy, `resample` geometry)\n\n"
            "| Control | Model | Metadata shortcut |\n|---|---|---|\n"
            f"| Resolution-matched subset | {rm['model']['balanced_accuracy']:.3f} | "
            f"{rm['shortcut_size_and_intensity']['balanced_accuracy']:.3f} |\n"
            f"| Train 640x480, test 1280x960 | {a['model']['balanced_accuracy']:.3f} | "
            f"{a['shortcut_size_and_intensity']['balanced_accuracy']:.3f} |\n"
            f"| Train 1280x960, test 640x480 | {b['model']['balanced_accuracy']:.3f} | "
            f"{b['shortcut_size_and_intensity']['balanced_accuracy']:.3f} |\n"
            f"| Background only (particles masked) | {t['background_only']['balanced_accuracy']:.3f} | |\n")
    return "\n\n".join(p for p in parts if p)


def build_app() -> gr.Blocks:
    ex = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
    with gr.Blocks(title="Microplastic analysis") as demo:
        gr.Markdown("# Microplastic analysis\nPolymer from a fluorescence image; concentration "
                    "from four photodiode counts.")
        with gr.Tab("Polymer"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="filepath", label="Fluorescence image (4:3)")
                    btn = gr.Button("Analyse", variant="primary")
                    if ex:
                        gr.Examples(ex, inputs=img)
                with gr.Column():
                    txt = gr.Markdown()
                    overlay = gr.Image(label="Particle outlines")
                    hist = gr.Plot(label="Size histogram")
            btn.click(polymer_tab, img, [txt, overlay, hist], api_name="polymer")
        with gr.Tab("Concentration"):
            with gr.Row():
                with gr.Column():
                    inputs = [gr.Number(label=f"S{i}", value=v, precision=0)
                              for i, v in zip(range(1, 5), (160, 70, 64, 64))]
                    adc = gr.Dropdown(["mcp3008", "ads1115"], value="mcp3008", label="ADC")
                    b2 = gr.Button("Estimate", variant="primary")
                with gr.Column():
                    out = gr.Markdown()
                    plot = gr.Plot()
            b2.click(concentration_tab, inputs + [adc], [out, plot], api_name="concentration")
        with gr.Tab("About"):
            gr.Markdown(about_text())
    return demo


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", default="false", choices=["true", "false"])
    a = ap.parse_args(argv)
    build_app().launch(server_name="127.0.0.1", server_port=a.port, share=a.share == "true")


if __name__ == "__main__":
    main()
