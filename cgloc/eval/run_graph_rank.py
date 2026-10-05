"""E1b: BM25 vs graph-propagated BM25 (PPR) entity localization. Hyper-parameters chosen on the dev split."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from codegraph import build_graph
from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.gold import gold_locations
from cgloc.data.swebench import load_instances
from cgloc.eval.bm25 import rank_files, score_entities
from cgloc.eval.graph_rank import entity_adjacency, mix, seeded_ppr
from cgloc.eval.metrics import aggregate, rank_metrics

TOP_FILES = 10
GRID = {"top_k": (5, 10, 20, 50), "beta": (0.1, 0.2, 0.3, 0.5, 0.7), "restart": (0.3, 0.5)}


def cfg_key(c: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in c.items())


def evaluate(insts: list[dict], cache_dir: Path) -> dict[str, list]:
    results: dict[str, list] = defaultdict(list)
    parse_cache: dict[str, dict] = defaultdict(dict)
    for n, inst in enumerate(insts, 1):
        clone = ensure_clone(inst["repo"], cache_dir)
        ensure_commit(clone, inst["base_commit"])
        sources = read_files(clone, inst["base_commit"])
        gold = gold_locations(inst["patch"], sources.get)
        if not gold.entities:
            continue
        q = inst["problem_statement"]
        bm25 = score_entities(q, sources, files=rank_files(q, sources)[:TOP_FILES])
        adj, entities = entity_adjacency(build_graph(sources, cache=parse_cache[inst["repo"]]))
        results["bm25"].append(rank_metrics(sorted(bm25, key=lambda i: -bm25[i]), gold.entities))
        for top_k, restart in itertools.product(GRID["top_k"], GRID["restart"]):
            ppr = seeded_ppr(bm25, adj, top_k, restart)
            for beta in GRID["beta"]:
                key = cfg_key({"top_k": top_k, "beta": beta, "restart": restart})
                results[key].append(rank_metrics(mix(bm25, ppr, entities, beta), gold.entities))
        print(f"[{n}/{len(insts)}] {inst['instance_id']}", flush=True)
    return results


def evaluate_fixed(insts: list[dict], cache_dir: Path, top_k: int, beta: float, restart: float) -> list[dict]:
    """Per-instance BM25 vs BM25+PPR metrics for one fixed configuration."""
    rows = []
    parse_cache: dict[str, dict] = defaultdict(dict)
    for n, inst in enumerate(insts, 1):
        clone = ensure_clone(inst["repo"], cache_dir)
        ensure_commit(clone, inst["base_commit"])
        sources = read_files(clone, inst["base_commit"])
        gold = gold_locations(inst["patch"], sources.get)
        if not gold.entities:
            continue
        q = inst["problem_statement"]
        bm25 = score_entities(q, sources, files=rank_files(q, sources)[:TOP_FILES])
        adj, entities = entity_adjacency(build_graph(sources, cache=parse_cache[inst["repo"]]))
        ppr_rank = mix(bm25, seeded_ppr(bm25, adj, top_k, restart), entities, beta)
        rows.append({"instance_id": inst["instance_id"], "repo": inst["repo"],
                     "bm25": rank_metrics(sorted(bm25, key=lambda i: -bm25[i]), gold.entities),
                     "ppr": rank_metrics(ppr_rank, gold.entities), "pred_ppr": ppr_rank[:20]})
        print(f"[{n}/{len(insts)}] {inst['instance_id']}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--out", default="outputs/graph_rank_lite.json")
    ap.add_argument("--config", help="evaluate one fixed setting on test, e.g. top_k=50,beta=0.5,restart=0.5")
    args = ap.parse_args()

    if args.config:
        from cgloc.eval.agent_report import bootstrap

        cfg = dict(kv.split("=") for kv in args.config.split(","))
        rows = evaluate_fixed(load_instances(args.dataset, "test"), Path(args.cache_dir),
                              int(cfg["top_k"]), float(cfg["beta"]), float(cfg["restart"]))
        Path(args.out).with_suffix(".per_instance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = {"config": args.config, "n": len(rows),
                  "bm25": aggregate(r["bm25"] for r in rows), "ppr": aggregate(r["ppr"] for r in rows),
                  "paired_bootstrap_ppr_minus_bm25": {
                      k: dict(zip(("diff", "ci_lo", "ci_hi", "p"),
                                  bootstrap([r["ppr"][k] for r in rows], [r["bm25"][k] for r in rows])))
                      for k in ("acc@1", "acc@3", "acc@5", "acc@10", "mrr")}}
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report["paired_bootstrap_ppr_minus_bm25"], indent=2))
        return

    dev = {k: aggregate(v) for k, v in evaluate(load_instances(args.dataset, "dev"), Path(args.cache_dir)).items()}
    best = max((k for k in dev if k != "bm25"), key=lambda k: (dev[k]["acc@5"], dev[k]["mrr"]))
    print("dev bm25:", dev["bm25"], "\ndev best:", best, dev[best], flush=True)
    test = {k: aggregate(v) for k, v in evaluate(load_instances(args.dataset, "test"), Path(args.cache_dir)).items()}
    report = {"selected_on_dev": best, "dev": dev, "test": test,
              "test_bm25": test["bm25"], "test_selected": test[best],
              "test_oracle_best": max((k for k in test if k != "bm25"), key=lambda k: test[k]["acc@5"])}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("selected_on_dev", "test_bm25", "test_selected", "test_oracle_best")},
                     indent=2))


if __name__ == "__main__":
    main()
