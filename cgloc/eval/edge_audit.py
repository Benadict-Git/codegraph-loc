"""Audit codegraph call edges against Jedi's static `goto` on a stratified random sample."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import jedi

from codegraph import build_graph
from codegraph.parse import _parser
from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.swebench import load_instances


def call_sites(src: bytes, start: int, end: int, name: str) -> list[tuple[int, int]]:
    """(line, col) of every call to `name` whose callee identifier lies within lines [start, end] (1-based)."""
    out, stack = [], [_parser.parse(src).root_node]
    while stack:
        node = stack.pop()
        if node.end_point[0] + 1 < start or node.start_point[0] + 1 > end:
            continue
        if node.type == "call":
            fn = node.child_by_field_name("function")
            ident = fn.child_by_field_name("attribute") if fn is not None and fn.type == "attribute" else fn
            if ident is not None and ident.type == "identifier" and ident.text.decode() == name:
                out.append((ident.start_point[0] + 1, ident.start_point[1]))
        stack.extend(node.children)
    return out


def judge(project: jedi.Project, root: Path, sources: dict[str, bytes], nodes: dict, src_id: str, dst_id: str) -> str:
    s, d = nodes[src_id], nodes[dst_id]
    sites = call_sites(sources[s["file"]], s["start"], s["end"], d["name"])
    if not sites:
        return "no_call_site"
    code = sources[s["file"]].decode("utf-8", "replace")
    script = jedi.Script(code, path=str(root / s["file"]), project=project)
    # Jedi may answer with a bundled type stub (e.g. django-stubs); compare fully qualified names as well.
    dst_full = f"{nodes[d['file']].get('module', '')}.{dst_id.split('::', 1)[1]}"
    resolved = False
    for line, col in sites:
        try:
            targets = script.goto(line, col, follow_imports=True)
        except Exception:
            continue
        for t in targets:
            if t.full_name is None and (t.module_path is None or t.line is None):
                continue
            resolved = True
            if t.full_name == dst_full:
                return "agree"
            if t.module_path is None or t.line is None:
                continue
            try:
                rel = Path(t.module_path).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if rel == d["file"] and d["start"] <= t.line <= d["end"]:
                return "agree"
    return "disagree" if resolved else "jedi_unresolved"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=75)
    ap.add_argument("--repos", type=int, default=12, help="one Lite snapshot per repository, up to this many")
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--work-dir", default="cache/edge_audit")
    ap.add_argument("--out", default="outputs/edge_audit.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    first = {}
    for inst in load_instances("lite"):
        first.setdefault(inst["repo"], inst)
    pool: dict[str, list] = defaultdict(list)
    snaps = {}
    for repo, inst in list(sorted(first.items()))[:args.repos]:
        clone = ensure_clone(repo, args.cache_dir)
        ensure_commit(clone, inst["base_commit"])
        sources = read_files(clone, inst["base_commit"])
        root = Path(args.work_dir) / inst["instance_id"]
        for path, data in sources.items():
            f = root / path
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                f.write_bytes(data)
        graph = build_graph(sources)
        nodes = {n["id"]: n for n in graph["nodes"]}
        snaps[inst["instance_id"]] = (root, sources, nodes, jedi.Project(path=str(root)))
        for s, d, kind, how in graph["edges"]:
            if kind == "calls" and not nodes[s]["test"]:
                pool[how].append((inst["instance_id"], s, d))
        print(f"{inst['instance_id']}: {sum(len(v) for v in pool.values())} pooled call edges", flush=True)

    rng = random.Random(args.seed)
    results, rows = defaultdict(Counter), []
    for how in sorted(pool):
        for iid, s, d in rng.sample(pool[how], min(args.per_label, len(pool[how]))):
            root, sources, nodes, project = snaps[iid]
            verdict = judge(project, root, sources, nodes, s, d)
            results[how][verdict] += 1
            rows.append({"instance_id": iid, "resolution": how, "src": s, "dst": d, "verdict": verdict})
        c = results[how]
        print(how, dict(c), flush=True)

    summary = {}
    for how, c in results.items():
        judged = c["agree"] + c["disagree"]
        summary[how] = {**c, "precision_vs_jedi": round(c["agree"] / judged, 3) if judged else None,
                        "pooled_edges": len(pool[how])}
    Path(args.out).write_text(json.dumps({"summary": summary, "edges": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
