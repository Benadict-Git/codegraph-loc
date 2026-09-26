"""Graph-propagated lexical retrieval: personalized PageRank seeded by BM25 entity scores."""
from __future__ import annotations

from collections import defaultdict

_EDGE_KINDS = ("calls", "inherits", "contains")


def entity_adjacency(graph: dict) -> tuple[dict[str, set[str]], set[str]]:
    """Undirected adjacency over non-test files and non-nested entities; nested defs merge into their owner."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    parent = {dst: src for src, dst, kind, _ in graph["edges"] if kind == "contains"}

    def owner(nid: str) -> str:
        while nodes[nid].get("nested") and nid in parent:
            nid = parent[nid]
        return nid

    keep = {i for i, n in nodes.items() if not n["test"]}
    adj: dict[str, set[str]] = defaultdict(set)
    for src, dst, kind, _ in graph["edges"]:
        if kind not in _EDGE_KINDS or src not in keep or dst not in keep:
            continue
        a, b = owner(src), owner(dst)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    entities = {i for i in keep if nodes[i]["kind"] != "file" and not nodes[i].get("nested")}
    return adj, entities


def personalized_pagerank(adj: dict[str, set[str]], seeds: dict[str, float], restart: float = 0.3,
                          iters: int = 20) -> dict[str, float]:
    total = sum(seeds.values())
    if total <= 0:
        return {}
    p0 = {k: v / total for k, v in seeds.items()}
    p = dict(p0)
    for _ in range(iters):
        nxt: dict[str, float] = defaultdict(float)
        for k, v in p0.items():
            nxt[k] += restart * v
        for u, mass in p.items():
            nbrs = adj.get(u)
            if not nbrs:
                nxt[u] += (1 - restart) * mass
                continue
            share = (1 - restart) * mass / len(nbrs)
            for v in nbrs:
                nxt[v] += share
        p = nxt
    return p


def seeded_ppr(bm25: dict[str, float], adj: dict[str, set[str]], top_k: int, restart: float) -> dict[str, float]:
    ranked = sorted(bm25, key=lambda i: -bm25[i])
    return personalized_pagerank(adj, {i: max(bm25[i], 0.0) for i in ranked[:top_k]}, restart)


def graph_rerank(bm25: dict[str, float], adj: dict[str, set[str]], entities: set[str],
                 top_k: int = 20, beta: float = 0.5, restart: float = 0.3) -> list[str]:
    """score = (1-beta) * bm25/max + beta * ppr/max, PPR seeded with the top_k BM25 entities."""
    return mix(bm25, seeded_ppr(bm25, adj, top_k, restart), entities, beta)


def mix(bm25: dict[str, float], ppr: dict[str, float], entities: set[str], beta: float) -> list[str]:
    bmax = max(bm25.values(), default=0.0) or 1.0
    pmax = max((v for k, v in ppr.items() if k in entities), default=0.0) or 1.0
    cands = (set(bm25) | set(ppr)) & entities
    score = {c: (1 - beta) * max(bm25.get(c, 0.0), 0.0) / bmax + beta * ppr.get(c, 0.0) / pmax for c in cands}
    return sorted(cands, key=lambda c: (-score[c], c))
