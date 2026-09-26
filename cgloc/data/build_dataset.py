"""Build the CodeGraph-Loc dataset: one code graph + gold locations per SWE-bench instance."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from codegraph import build_graph, save_graph
from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.gold import gold_locations
from cgloc.data.swebench import load_instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--out", default="outputs/dataset/lite")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    insts = load_instances(args.dataset, args.split)
    by_repo = defaultdict(list)
    for inst in insts:
        by_repo[inst["repo"]].append(inst)

    stats = []
    with (out / "instances.jsonl").open("w") as f:
        for repo, group in sorted(by_repo.items()):
            clone = ensure_clone(repo, args.cache_dir)
            cache: dict = {}
            for inst in group:
                t0 = time.time()
                ensure_commit(clone, inst["base_commit"])
                sources = read_files(clone, inst["base_commit"])
                gold = gold_locations(inst["patch"], sources.get)
                graph = build_graph(sources, cache=cache, meta={
                    "instance_id": inst["instance_id"], "repo": repo, "base_commit": inst["base_commit"]})
                save_graph(graph, out / "graphs" / f"{inst['instance_id']}.json.gz")
                edges = Counter(e[2] for e in graph["edges"])
                calls = Counter(e[3] for e in graph["edges"] if e[2] == "calls")
                row = {
                    "instance_id": inst["instance_id"],
                    "repo": repo,
                    "base_commit": inst["base_commit"],
                    "problem_statement": inst["problem_statement"],
                    "gold": asdict(gold),
                }
                f.write(json.dumps(row) + "\n")
                stats.append({"instance_id": inst["instance_id"], "repo": repo, "files": len(sources),
                              "nodes": len(graph["nodes"]), **{f"edges_{k}": v for k, v in edges.items()},
                              **{f"calls_{k}": v for k, v in calls.items()}, "secs": round(time.time() - t0, 2)})
                print(f"{inst['instance_id']}: {len(graph['nodes'])} nodes, {sum(edges.values())} edges "
                      f"({stats[-1]['secs']}s)", flush=True)
    (out / "stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))


if __name__ == "__main__":
    main()
