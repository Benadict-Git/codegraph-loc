"""E1 baseline: BM25 file- and entity-level localization on SWE-bench."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.gold import gold_locations
from cgloc.data.swebench import load_instances
from cgloc.eval.bm25 import rank_entities, rank_files
from cgloc.eval.metrics import aggregate, rank_metrics

ENTITY_TOP_FILES = 10


def run_instance(inst: dict, cache_dir: Path) -> dict:
    t0 = time.time()
    clone = ensure_clone(inst["repo"], cache_dir)
    ensure_commit(clone, inst["base_commit"])
    sources = read_files(clone, inst["base_commit"])
    gold = gold_locations(inst["patch"], sources.get)
    files = rank_files(inst["problem_statement"], sources)
    entities = rank_entities(inst["problem_statement"], sources, files=files[:ENTITY_TOP_FILES])
    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "gold": asdict(gold),
        "pred_files": files[:50],
        "pred_entities": entities[:50],
        "file": rank_metrics(files, gold.files),
        "entity": rank_metrics(entities, gold.entities),
        "n_files": len(sources),
        "secs": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--repos", nargs="*", help="only these repos, e.g. psf/requests")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--out", default="outputs/bm25_lite.jsonl")
    args = ap.parse_args()

    insts = load_instances(args.dataset)
    if args.repos:
        insts = [i for i in insts if i["repo"] in args.repos]
    insts = insts[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as f:
        for n, inst in enumerate(insts, 1):
            row = run_instance(inst, Path(args.cache_dir))
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"[{n}/{len(insts)}] {row['instance_id']} file_acc@5={row['file'] and row['file']['acc@5']}"
                  f" ({row['secs']}s)", flush=True)
    summary = {"file": aggregate(r["file"] for r in rows), "entity": aggregate(r["entity"] for r in rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
