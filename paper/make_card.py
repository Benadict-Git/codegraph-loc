"""Render the 560x280 Kaggle card (drawn at 2x) from the measured results.

The left 280x280 square is self-contained because Kaggle crops it as the square thumbnail.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 560 * S, 280 * S
SURFACE, TEXT, TEXT_2, MUTED, TRACK = "#1a1a19", "#ffffff", "#c3c2b7", "#8f8e86", "#2c2c2a"
COLOR_BM25, COLOR_E4B, COLOR_12B = "#6e7681", "#388bfd", "#2ea043"
FONT = "/usr/share/fonts/noto/NotoSans-{}.ttf"


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size * S)


def wrap(d, text, fnt, width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


rows = [
    ("BM25 lexical baseline", 0.197, COLOR_BM25),
    ("Gemma 4 E4B + graph", 0.328, COLOR_E4B),
    ("Gemma 4 12B + graph", 0.457, COLOR_12B),
]

img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)

x0 = 24 * S
d.text((x0, 22 * S), "CodeGraph-Loc", font=font("Bold", 23), fill=TEXT)
bar_w, bar_h = 155 * S, 10 * S
y = 66 * S
for label, value, color in rows:
    d.text((x0, y), label, font=font("Medium", 11), fill=TEXT_2)
    by = y + 18 * S
    d.rounded_rectangle((x0, by, x0 + bar_w, by + bar_h), radius=4 * S, fill=TRACK)
    d.rounded_rectangle((x0, by, x0 + int(bar_w * value), by + bar_h), radius=4 * S, fill=color)
    d.text((x0 + bar_w + 8 * S, by - 6 * S), f"{value * 100:.0f}%", font=font("Bold", 16), fill=TEXT)
    y += 48 * S

d.text((x0, 228 * S), "Top-1 function localization, SWE-bench Lite", font=font("Regular", 9), fill=MUTED)

d.line((280 * S, 28 * S, 280 * S, 252 * S), fill=TRACK, width=S)
rx, rw = 300 * S, 236 * S
ty = 32 * S
title_text = "Graph-guided bug localization with Gemma 4 on consumer GPUs"
for line in wrap(d, title_text, font("Medium", 16), rw):
    d.text((rx, ty), line, font=font("Medium", 16), fill=TEXT)
    ty += 24 * S
ty += 12 * S
items = (
    "785 SWE-bench Lite & Verified instances",
    "Laptop-scale tree-sitter graphs (3s/repo)",
    "Function-level gold supervision",
    "Gemma 4 12B reaches 45.7% top-1 accuracy",
    "Audit of competition tools on free T4",
)
for item in items:
    d.text((rx, ty), f"·  {item}", font=font("Regular", 11), fill=TEXT_2)
    ty += 21 * S

out = Path("paper/card")
out.mkdir(exist_ok=True)
img.save(out / "card_1120x560.png")
img.resize((560, 280), Image.LANCZOS).save(out / "card_560x280.png")
print("wrote", *sorted(p.name for p in out.iterdir()))
