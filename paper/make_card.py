"""Render the 560x280 Kaggle card (drawn at 2x) from the measured SWE-bench Lite BM25 results."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 560 * S, 280 * S
SURFACE, TEXT, TEXT_2, MUTED, TRACK, SERIES = "#1a1a19", "#ffffff", "#c3c2b7", "#8f8e86", "#2c2c2a", "#3987e5"
FONT = "/usr/share/fonts/noto/NotoSans-{}.ttf"


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size * S)


summary = json.loads(Path("outputs/bm25_lite.summary.json").read_text())
rows = [("Right file in top 5", summary["file"]["acc@5"]),
        ("All right functions in top 5", summary["entity"]["acc@5"])]

img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)
x0 = 32 * S
d.text((x0, 26 * S), "CodeGraph-Loc", font=font("Bold", 30), fill=TEXT)
d.text((x0, 66 * S), "Graph-guided bug localization for small local agents", font=font("Regular", 15), fill=TEXT_2)

bar_x, bar_w, bar_h = x0, 400 * S, 14 * S
y = 118 * S
for label, value in rows:
    d.text((bar_x, y), label, font=font("Medium", 14), fill=TEXT_2)
    by = y + 24 * S
    d.rounded_rectangle((bar_x, by, bar_x + bar_w, by + bar_h), radius=4 * S, fill=TRACK)
    d.rounded_rectangle((bar_x, by, bar_x + int(bar_w * value), by + bar_h), radius=4 * S, fill=SERIES)
    d.text((bar_x + bar_w + 14 * S, by - 9 * S), f"{value * 100:.0f}%", font=font("Bold", 22), fill=TEXT)
    y += 58 * S

d.text((x0, 244 * S), "BM25 baseline on SWE-bench Lite test  ·  tree-sitter graphs  ·  runs on a free T4",
       font=font("Regular", 11), fill=MUTED)

out = Path("paper/card")
out.mkdir(exist_ok=True)
img.save(out / "card_1120x560.png")
img.resize((560, 280), Image.LANCZOS).save(out / "card_560x280.png")
print("wrote", *sorted(p.name for p in out.iterdir()))
