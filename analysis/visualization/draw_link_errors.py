"""
Overlay entity-linking results (TP / FP / FN) on a TONDFU test image, using the dump
written by analysis/tondfu_eval_dump.py. No model inference needed.

    green  = TP (correct link)
    red    = FP (predicted, not true)
    orange = FN (true, missed)
Entity boxes are blue; boxes with a wrong predicted class are drawn magenta.
Link endpoints: green dot = key (src), red dot = value (dst).

Usage (from repo root):
    python analysis/visualization/draw_link_errors.py --doc IMG_2703 \
        --dump outputs/tondfu_error_analysis/e2e_char_embed-20250508-0136-6d89b3
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.paths import TONDFU_TEST
from analysis.visualization.draw import draw_results

STATUS_STYLE = {'TP': ('limegreen', 5), 'FN': ('orange', 5), 'FP': ('red', 5)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--doc', required=True)
    p.add_argument('--dump', required=True, help='folder with nodes.json and links.json')
    args = p.parse_args()

    dump = Path(args.dump)
    nodes = [n for n in json.load(open(dump / 'nodes.json')) if n['doc'] == args.doc]
    links = [l for l in json.load(open(dump / 'links.json')) if l['doc'] == args.doc]
    boxes = [n['box'] for n in sorted(nodes, key=lambda n: n['node'])]

    img = Image.open(TONDFU_TEST / 'images' / f'{args.doc}.jpg').convert('RGB')

    # entity boxes, highlighting mislabeled ones
    draw = ImageDraw.Draw(img)
    for n in nodes:
        draw.rectangle(n['box'], outline='blue' if n['true'] == n['pred'] else 'magenta', width=3)

    for status in ['TP', 'FN', 'FP']:
        sel = [l for l in links if l['status'] == status]
        u = [l['src'] for l in sel]
        v = [l['dst'] for l in sel]
        color, width = STATUS_STYLE[status]
        draw_results(img, boxes, list(range(len(sel))), u, v,
                     line_color=color, line_width=width, draw_boxes=False)

    counts = {s: sum(l['status'] == s for l in links) for s in STATUS_STYLE}
    draw.rectangle([0, 0, 900, 50], fill='white')
    draw.text((10, 10), f"{args.doc}  TP(green)={counts['TP']}  FP(red)={counts['FP']}  "
                        f"FN(orange)={counts['FN']}  magenta box = wrong class", fill='black')

    out = dump / 'figures' / f'{args.doc}.jpg'
    out.parent.mkdir(exist_ok=True)
    img.save(out, quality=90)
    print(f'-> saved {out}')


if __name__ == '__main__':
    main()
