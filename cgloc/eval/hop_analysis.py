"""How far was the gold entity from code the agent actually looked at? (one-hop opportunity analysis)"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

from cgloc.agent.run import owner_entity, prepare
from cgloc.data.swebench import load_instances
from cgloc.eval.graph_rank import entity_adjacency
from codegraph import CodeGraph, load_graph

_GREP_HIT = re.compile(r"^([\w./\-]+\.py):(\d+): ", re.M)
READ_LINES = 120


def seen_entities(r: dict, cg) -> set[str]:
    """Entities whose code was shown to the agent via read_file, read_entity or grep hits."""
    seen: set[str] = set()
    obs = [m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].startswith("Observation:")]
    tool_steps = [s for s in r["steps"] if "tool" in s]
    for step, text in zip(tool_steps, obs):
        args = step.get("args", {})
        if step["tool"] == "read_file" and cg.get(str(args.get("path", ""))):
            path = args["path"]
            try:
                start = max(1, int(args.get("start_line", 1) or 1))
                end = int(args.get("end_line") or start + READ_LINES - 1)
            except (TypeError, ValueError):
                continue
            end = min(end, start + READ_LINES - 1)
            seen.update(n["id"] for n in cg.nodes.values() if n["file"] == path and n["kind"] != "file"
                        and not n.get("nested") and n["start"] <= end and n["end"] >= start)
        elif step["tool"] == "read_entity":
            first = text.split("\n", 2)[1] if "\n" in text else ""
            if cg.get(first):
                seen.add(owner_entity(cg, first))
        elif step["tool"] == "grep":
            for path, line in _GREP_HIT.findall(text):
                nid = cg.locate(path, int(line)) if cg.get(path) else None
                if nid and cg.get(nid)["kind"] != "file":
                    seen.add(owner_entity(cg, nid))
    return {e for e in seen if cg.get(e) and cg.get(e)["kind"] != "file"}


def distance(adj: dict[str, set[str]], sources: set[str], targets: set[str], limit: int = 2) -> int | None:
    """Shortest undirected hop count from any source to any target, up to `limit`."""
    frontier, dist = deque(sources), {s: 0 for s in sources}
    while frontier:
        cur = frontier.popleft()
        if cur in targets:
            return dist[cur]
        if dist[cur] < limit:
            for nxt in adj.get(cur, ()):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    frontier.append(nxt)
    return None


def _get_cg(iid: str, inst: dict, cache_dir: Path, dataset: str, cg_cache: dict[str, CodeGraph]) -> CodeGraph:
    if iid in cg_cache:
        return cg_cache[iid]
    for ext in (".json.gz", ".json"):
        p = Path(f"outputs/dataset/{dataset}/graphs/{iid}{ext}")
        if p.exists():
            cg = CodeGraph(load_graph(p))
            cg_cache[iid] = cg
            return cg
    _, cg, _ = prepare(inst, cache_dir)
    cg_cache[iid] = cg
    return cg


def analyze(run_path: Path, insts: dict[str, dict], cache_dir: Path, dataset: str = "lite", cg_cache: dict[str, CodeGraph] | None = None) -> dict:
    if cg_cache is None:
        cg_cache = {}
    cats: Counter = Counter()
    for r in map(json.loads, run_path.read_text().splitlines()):
        gold = set(r["gold"]["entities"])
        if not gold:
            continue
        if gold & set(r["pred_entities"][:5]):
            cats["hit (gold in answer)"] += 1
            continue
        cg = _get_cg(r["instance_id"], insts[r["instance_id"]], cache_dir, dataset, cg_cache)
        seen = seen_entities(r, cg)
        anchors = seen | set(r["pred_entities"])
        if gold & seen:
            cats["miss: gold code was shown"] += 1
            continue
        adj, _ = entity_adjacency({"nodes": list(cg.nodes.values()),
                                   "edges": [[s, d, k, ""] for s, outs in cg.out.items() for d, k in outs]})
        d = distance(adj, anchors, gold) if anchors else None
        if d == 1:
            cats["miss: gold 1 hop from seen/answered"] += 1
        elif d == 2:
            cats["miss: gold 2 hops from seen/answered"] += 1
        elif any(cg.get(g) and cg.get(g)["file"] in {cg.get(a)["file"] for a in anchors} for g in gold):
            cats["miss: same file, >2 hops"] += 1
        else:
            cats["miss: elsewhere"] += 1
    n = sum(cats.values())
    return {"n": n, **{k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in sorted(cats.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path.jsonl (rescored agent runs)")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--out", default="outputs/hop_analysis.json")
    args = ap.parse_args()
    insts = {i["instance_id"]: i for i in load_instances(args.dataset)}
    cg_cache: dict[str, CodeGraph] = {}
    report = {name: analyze(Path(path), insts, Path(args.cache_dir), dataset=args.dataset, cg_cache=cg_cache)
              for name, path in (s.split("=", 1) for s in args.runs)}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
