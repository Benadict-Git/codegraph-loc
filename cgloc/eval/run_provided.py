"""E6: audit the competition-provided code graphs and embeddings as localization signals.

Runs on the competition's public dev tasks (tasks.jsonl) with the provided graphs/<repo>_<commit>.json and
embeddings/<repo>_<commit>.npz, compared against BM25 and codegraph under identical scoring.
The graph-propagation hyper-parameters are taken from the SWE-bench Lite dev selection (no tuning here).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from codegraph import build_graph
from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.gold import gold_locations
from cgloc.eval.bm25 import rank_files, score_entities
from cgloc.eval.graph_rank import entity_adjacency, mix, seeded_ppr
from cgloc.eval.metrics import aggregate, rank_metrics

TOP_FILES = 10
RRF_K = 60
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def load_provided_graph(path: Path) -> tuple[list[str], list[tuple[str, str, str]]]:
    data = json.loads(path.read_text())
    nodes = [n["id"] for n in data["nodes"]]
    edges = [(e["source"], e["target"], str(e.get("type", ""))) for e in data.get("edges", data.get("links", []))]
    return nodes, edges


def map_to_entities(provided: list[str], entities: set[str]) -> dict[str, str]:
    """Map provided dotted symbol ids to codegraph entity ids by longest dotted-suffix match."""
    known = set(provided)
    out = {}
    for eid in entities:
        path, qual = eid.split("::", 1)
        full = (path[:-3].removesuffix("/__init__").replace("/", ".") + "." + qual).split(".")
        for i in range(len(full) - len(qual.split(".")) + 1):
            pid = ".".join(full[i:])
            if pid in known and pid not in out:
                out[pid] = eid
                break
    return out


def provided_adjacency(edges, p2e: dict[str, str]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for s, t, _ in edges:
        a, b = p2e.get(s), p2e.get(t)
        if a and b and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def resolve_symbol(query: str, keys: list[str], lower: dict[str, list[str]]) -> str | None:
    """The harness's 4-tier lookup: exact, dotted suffix, case-insensitive, substring."""
    if query in lower.get("__exact__", ()):
        return query
    suffix = [k for k in keys if k.endswith("." + query)]
    if suffix:
        return min(suffix, key=len)
    ci = lower.get(query.lower())
    if ci:
        return ci[0]
    sub = [k for k in keys if query in k]
    return min(sub, key=len) if sub else None


def rrf(*rankings: list[str]) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, x in enumerate(r):
            score[x] += 1.0 / (RRF_K + i + 1)
    return sorted(score, key=lambda x: (-score[x], x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", default="cache/kaggle/main")
    ap.add_argument("--tasks", help="tasks .jsonl (default: <comp-dir>/tasks.jsonl)")
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--selection", default="outputs/graph_rank_lite.json")
    ap.add_argument("--config", help="override, e.g. top_k=50,beta=0.5,restart=0.5 (as selected on dev)")
    ap.add_argument("--out", default="outputs/provided_audit.json")
    args = ap.parse_args()

    comp = Path(args.comp_dir)
    sel = args.config or json.loads(Path(args.selection).read_text())["selected_on_dev"]
    cfg = dict(kv.split("=") for kv in sel.split(","))
    top_k, beta, restart = int(cfg["top_k"]), float(cfg["beta"]), float(cfg["restart"])
    tasks = [json.loads(line) for line in Path(args.tasks or comp / "tasks.jsonl").read_text().splitlines() if line.strip()]

    metrics: dict[str, list] = defaultdict(list)
    stats = []
    per_task = []
    parse_cache: dict[str, dict] = defaultdict(dict)
    for n, t in enumerate(tasks, 1):
        short = t["repo"].split("/")[1]
        gpath = comp / "graphs" / f"{short}_{t['base_commit']}.json"
        epath = comp / "embeddings" / f"{short}_{t['base_commit']}.npz"
        clone = ensure_clone(t["repo"], args.cache_dir)
        ensure_commit(clone, t["base_commit"])
        sources = read_files(clone, t["base_commit"])
        gold = gold_locations(t["patch"], sources.get)
        q = t["problem_statement"]
        files = rank_files(q, sources)
        metrics["file_bm25"].append(rank_metrics(files, [f for f in gold.files if f.endswith(".py")]))
        if not gold.entities:
            continue

        ours = build_graph(sources, cache=parse_cache[t["repo"]])
        adj, entities = entity_adjacency(ours)
        bm25 = score_entities(q, sources, files=files[:TOP_FILES])
        bm25_rank = sorted(bm25, key=lambda i: -bm25[i])
        our_rank = mix(bm25, seeded_ppr(bm25, adj, top_k, restart), entities, beta)

        pnodes, pedges = load_provided_graph(gpath)
        p2e = map_to_entities(pnodes, entities)
        padj = provided_adjacency(pedges, p2e)
        prov_rank = mix(bm25, seeded_ppr(bm25, padj, top_k, restart), entities, beta)

        emb = np.load(epath)
        keys = [k for k in emb.keys() if k in p2e]
        mat = np.stack([emb[k] for k in keys]) if keys else np.zeros((0, 256), np.float32)
        mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        kidx = {k: i for i, k in enumerate(keys)}
        lower: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            lower[k.lower()].append(k)
        lower["__exact__"] = set(keys)

        seen, sym_rank = set(), []
        for sym in dict.fromkeys(_SYMBOL.findall(q)):
            if len(sym) < 4 or "." not in sym and not re.search(r"[A-Z_]", sym[1:]):
                continue
            key = resolve_symbol(sym, keys, lower)
            if key is None or key in seen:
                continue
            seen.add(key)
            sims = mat @ mat[kidx[key]]
            sym_rank.append([p2e[keys[i]] for i in np.argsort(-sims)[:10]])
        emb_symbol = list(dict.fromkeys(x for group in zip(*sym_rank) for x in group)) if sym_rank else []

        seeds = [kidx[p] for p, e in p2e.items() if e in set(bm25_rank[:5]) and p in kidx]
        if seeds:
            centroid = mat[seeds].mean(0)
            emb_centroid = [p2e[keys[i]] for i in np.argsort(-(mat @ centroid))]
        else:
            emb_centroid = []

        ranks = {
            "entity_bm25": bm25_rank,
            "entity_ppr_codegraph": our_rank,
            "entity_ppr_provided": prov_rank,
            "entity_emb_issue_symbols": emb_symbol,
            "entity_rrf_bm25_emb_centroid": rrf(bm25_rank, emb_centroid),
        }
        for k, r in ranks.items():
            metrics[k].append(rank_metrics(r, gold.entities))

        pe = Counter(e[2] for e in pedges)
        oe = Counter(e[2] for e in ours["edges"])
        prov_ids = set(p2e.values())
        stats.append({
            "instance_id": t["instance_id"], "repo": t["repo"],
            "provided_nodes": len(pnodes), "provided_edges": dict(pe), "provided_mapped": len(p2e),
            "provided_emb_keys": len(emb.keys()),
            "codegraph_nodes": len(ours["nodes"]), "codegraph_edges": dict(oe),
            "codegraph_entities": len(entities),
            "gold_entities": len(gold.entities),
            "gold_in_provided": sum(g in prov_ids for g in gold.entities),
            "gold_in_codegraph": sum(g in entities for g in gold.entities),
            "adj_edges_provided": sum(map(len, padj.values())) // 2,
            "adj_edges_codegraph": sum(map(len, adj.values())) // 2,
            "issue_symbols_resolved": len(sym_rank),
        })
        per_task.append({"instance_id": t["instance_id"], "gold": asdict(gold),
                         **{k: r[:20] for k, r in ranks.items()}})
        print(f"[{n}/{len(tasks)}] {t['instance_id']}", flush=True)

    report = {"config": {"top_k": top_k, "beta": beta, "restart": restart},
              "metrics": {k: aggregate(v) for k, v in metrics.items()}, "stats": stats}
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.out).with_suffix(".per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_task))
    for k, v in report["metrics"].items():
        print(k, {m: round(x, 3) for m, x in v.items() if m in ("acc@1", "acc@5", "acc@10", "mrr", "n")})


if __name__ == "__main__":
    main()
