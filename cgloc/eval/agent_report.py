"""§4.4 tables: agent runs vs BM25 on shared instances, with paired bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

KEYS = (("entity", "acc@1"), ("entity_backfill", "acc@5"), ("entity_backfill", "acc@10"))


def load(path: str | Path) -> dict[str, dict]:
    return {r["instance_id"]: r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean paired difference a-b (in points) with 95% CI and two-sided p-value."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    s = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    p = min(1.0, 2 * min(sum(x <= 0 for x in s), sum(x >= 0 for x in s)) / n)
    return 100 * sum(d) / len(d), 100 * s[int(0.025 * n)], 100 * s[int(0.975 * n)], p


def report(bm25: dict[str, dict], runs: dict[str, dict[str, dict]]) -> str:
    ids = sorted(set(bm25).intersection(*runs.values()))
    ent = [i for i in ids if bm25[i]["entity"]]
    pct = lambda xs: 100 * sum(xs) / len(xs)
    lines = [f"shared instances: {len(ids)} (with entity gold: {len(ent)})", "",
             "| run | Acc@1 | Acc@5 +fill | Acc@10 +fill | File Acc@1 | Unanswered | Calls | Sec/issue |",
             "|---|---|---|---|---|---|---|---|",
             f"| BM25 | {pct([bm25[i]['entity']['acc@1'] for i in ent]):.1f} | "
             f"{pct([bm25[i]['entity']['acc@5'] for i in ent]):.1f} | {pct([bm25[i]['entity']['acc@10'] for i in ent]):.1f} | "
             f"{pct([bm25[i]['file']['acc@1'] for i in ids]):.1f} | - | - | - |"]
    for name, R in runs.items():
        rows = [R[i] for i in ids]
        lines.append(
            f"| {name} | {pct([R[i]['entity']['acc@1'] for i in ent]):.1f} | "
            f"{pct([R[i]['entity_backfill']['acc@5'] for i in ent]):.1f} | "
            f"{pct([R[i]['entity_backfill']['acc@10'] for i in ent]):.1f} | "
            f"{pct([r['file']['acc@1'] for r in rows if r['file']]):.1f} | {sum(not r['pred_entities'] for r in rows)} | "
            f"{sum(r['tool_calls'] for r in rows) / len(rows):.1f} | {sum(r['secs'] for r in rows) / len(rows):.0f} |")
    lines += ["", "paired bootstrap (points, 95% CI, p):"]
    names = list(runs)
    for key, k in KEYS:
        for n in names:
            o, lo, hi, p = bootstrap([runs[n][i][key][k] for i in ent], [bm25[i]["entity"][k] for i in ent])
            lines.append(f"  {n} - BM25, {key} {k}: {o:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.3f}")
        for a, b in zip(names, names[1:]):
            o, lo, hi, p = bootstrap([runs[b][i][key][k] for i in ent], [runs[a][i][key][k] for i in ent])
            lines.append(f"  {b} - {a}, {key} {k}: {o:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.3f}")
    lines += ["", "tool usage:"]
    for n, R in runs.items():
        use = Counter(s["tool"] for r in R.values() for s in r["steps"] if "tool" in s)
        lines.append(f"  {n}: {dict(use.most_common())}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", default="outputs/bm25_lite.jsonl")
    ap.add_argument("runs", nargs="+", help="name=path.jsonl (rescored agent runs), compared in the given order")
    args = ap.parse_args()
    runs = {name: load(path) for name, path in (s.split("=", 1) for s in args.runs)}
    print(report(load(args.bm25), runs))


if __name__ == "__main__":
    main()
