"""Write docs/hardware/wiring.svg: laser -> cuvette -> 4 photodiodes -> TIA -> ADC -> Pi."""
from pathlib import Path

W, H = 900, 420


def box(x, y, w, h, text, fill="#eef3fb"):
    lines = text.split("\n")
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#334" />']
    for i, t in enumerate(lines):
        ty = y + h / 2 + (i - (len(lines) - 1) / 2) * 15 + 4
        out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="12">{t}</text>')
    return "\n".join(out)


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334" stroke-width="1.4" '
            'marker-end="url(#a)" />')


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="sans-serif">',
    '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
    '<path d="M0,0 L8,4 L0,8 z" fill="#334"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    '<text x="450" y="24" text-anchor="middle" font-size="15">Scattering channel: signal chain</text>',
    box(20, 175, 90, 60, "650 nm\nlaser"),
    arrow(110, 205, 170, 205),
    box(170, 160, 90, 90, "cuvette\n(sample)", "#f7f2e6"),
]
ys = [60, 140, 220, 300]
angles = ["θ1", "θ2", "θ3", "θ4"]
for i, y in enumerate(ys):
    parts.append(arrow(260, 205, 330, y + 25))
    parts.append(box(330, y, 90, 50, f"BPW34 #{i + 1}\n{angles[i]}"))
    parts.append(arrow(420, y + 25, 470, y + 25))
    parts.append(box(470, y, 110, 50, f"TIA {i + 1}\nR_f ∥ C_f"))
    parts.append(arrow(580, y + 25, 640, y + 25))
parts += [
    box(640, 60, 100, 290, "external ADC\nMCP3008 (SPI)\nor ADS1115 (I²C)", "#e9f5ea"),
    arrow(740, 205, 790, 205),
    box(790, 175, 90, 60, "Raspberry\nPi 5"),
    '<text x="450" y="400" text-anchor="middle" font-size="11">V_i = (R·P_i + I_dark)·R_f + V_offset;  '
    'counts_i = floor(V_i / V_fs · 2^bits)</text>',
    "</svg>",
]
out = Path(__file__).with_name("wiring.svg")
out.write_text("\n".join(parts), encoding="utf8")
print(out)
