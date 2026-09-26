"""Render the 560x280 Kaggle card (drawn at 2x) from the measured SWE-bench Lite BM25 results.

The left 280x280 square is self-contained because Kaggle crops it as the square thumbnail.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 560 * S, 280 * S
SURFACE, TEXT, TEXT_2, MUTED, TRACK, SERIES = "#1a1a19", "#ffffff", "#c3c2b7", "#8f8e86", "#2c2c2a", "#3987e5"
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


summary = json.loads(Path("outputs/bm25_lite.summary.json").read_text())
rows = [("Right file in top 5", summary["file"]["acc@5"]),
        ("All right functions in top 5", summary["entity"]["acc@5"])]

img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)

x0 = 24 * S
d.text((x0, 26 * S), "CodeGraph-Loc", font=font("Bold", 25), fill=TEXT)
bar_w, bar_h = 160 * S, 12 * S
y = 88 * S
for label, value in rows:
    d.text((x0, y), label, font=font("Medium", 12), fill=TEXT_2)
    by = y + 22 * S
    d.rounded_rectangle((x0, by, x0 + bar_w, by + bar_h), radius=4 * S, fill=TRACK)
    d.rounded_rectangle((x0, by, x0 + int(bar_w * value), by + bar_h), radius=4 * S, fill=SERIES)
    d.text((x0 + bar_w + 10 * S, by - 8 * S), f"{value * 100:.0f}%", font=font("Bold", 20), fill=TEXT)
    y += 62 * S
d.text((x0, 238 * S), "BM25, SWE-bench Lite test", font=font("Regular", 10), fill=MUTED)

d.line((288 * S, 32 * S, 288 * S, 248 * S), fill=TRACK, width=S)
rx, rw = 308 * S, 228 * S
ty = 40 * S
for line in wrap(d, "Graph-guided bug localization for small, local coding agents", font("Medium", 17), rw):
    d.text((rx, ty), line, font=font("Medium", 17), fill=TEXT)
    ty += 26 * S
ty += 14 * S
for item in ("tree-sitter code graphs", "function-level gold locations", "audit of the provided graph tools",
             "runs on a free Kaggle T4"):
    d.text((rx, ty), f"·  {item}", font=font("Regular", 12), fill=TEXT_2)
    ty += 22 * S

out = Path("paper/card")
out.mkdir(exist_ok=True)
img.save(out / "card_1120x560.png")
img.resize((560, 280), Image.LANCZOS).save(out / "card_560x280.png")
img.crop((0, 0, H, H)).resize((280, 280), Image.LANCZOS).save(out / "thumb_preview.png")
print("wrote", *sorted(p.name for p in out.iterdir()))
