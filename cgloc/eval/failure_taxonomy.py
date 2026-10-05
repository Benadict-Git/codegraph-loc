"""Classify agent outcomes: hit / no usable answer / wrong file / right file but wrong function."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify(r: dict) -> str:
    gold = set(r["gold"]["entities"])
    gold_files = {g.split("::", 1)[0] for g in gold}
    if gold & set(r["pred_entities"][:5]):
        return "hit"
    if not r["pred_entities"]:
        forced = bool(r["steps"]) and r["steps"][-1].get("forced")
        if not r["raw_final"]:
            return "no answer: budget/format" if forced else "no answer: empty list"
        return "no answer: names not in repo"
    if gold_files & {e.split("::", 1)[0] for e in r["pred_entities"][:5]}:
        return "right file, wrong function"
    return "wrong file"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path.jsonl (rescored agent runs)")
    ap.add_argument("--out", default="outputs/failure_taxonomy.json")
    args = ap.parse_args()
    report = {}
    for name, path in (s.split("=", 1) for s in args.runs):
        rows = [r for r in map(json.loads, Path(path).read_text().splitlines()) if r["gold"]["entities"]]
        c = Counter(classify(r) for r in rows)
        report[name] = {"n": len(rows), **{k: {"count": v, "pct": round(100 * v / len(rows), 1)}
                                           for k, v in c.most_common()}}
    Path(args.out).write_text(json.dumps(report, indent=2))
    for name, rep in report.items():
        print(name, {k: v["pct"] for k, v in rep.items() if k != "n"}, "n =", rep["n"])


if __name__ == "__main__":
    main()
