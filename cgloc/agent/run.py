"""E2/E3: LLM localization agent with file tools only vs file + codegraph tools."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path

from codegraph import CodeGraph, build_graph
from cgloc.agent.llm import OpenAIChat
from cgloc.agent.loop import run_episode
from cgloc.agent.tools import Workspace, file_tools, graph_tools
from cgloc.data.checkout import ensure_clone, ensure_commit, read_files
from cgloc.data.gold import gold_locations
from cgloc.data.swebench import load_instances
from cgloc.eval.bm25 import rank_files
from cgloc.eval.metrics import aggregate, rank_metrics

CONDITIONS = ("files", "graph")


def owner_entity(cg: CodeGraph, nid: str) -> str:
    while cg.get(nid).get("nested") and cg.parent(nid):
        nid = cg.parent(nid)
    return nid


def normalize(answer: list[str], cg: CodeGraph) -> tuple[list[str], list[str]]:
    """Map free-form answers to (entity ids, file paths), preserving order."""
    ents, files = [], []
    for raw in answer:
        s = raw.strip().strip("`").split(" ")[0].replace(":L", "::").rstrip(",.")
        nid = None
        if cg.get(s) is not None:
            nid = s
        elif "::" in s:
            path, qual = s.split("::", 1)
            qual = qual.split("(")[0]
            if cg.get(f"{path}::{qual}") is not None:
                nid = f"{path}::{qual}"
            else:
                hits = [h["id"] for h in cg.search(qual, limit=20) if h["file"] == path]
                nid = hits[0] if hits else (path if cg.get(path) else None)
        elif s.endswith(".py"):
            nid = s if cg.get(s) else None
        else:
            hits = cg.search(s.split("(")[0], limit=1)
            nid = hits[0]["id"] if hits else None
        if nid is None:
            continue
        node = cg.get(nid)
        if node["kind"] == "file":
            files.append(nid)
            continue
        nid = owner_entity(cg, nid)
        ents.append(nid)
        files.append(cg.get(nid)["file"])
    return list(dict.fromkeys(ents)), list(dict.fromkeys(files))


def prepare(inst: dict, cache_dir: Path) -> tuple[dict, CodeGraph, Workspace]:
    clone = ensure_clone(inst["repo"], cache_dir)
    ensure_commit(clone, inst["base_commit"])
    sources = read_files(clone, inst["base_commit"])
    gold = gold_locations(inst["patch"], sources.get)
    ws = Workspace(sources)
    cg = CodeGraph(build_graph(sources), ws.text)
    return asdict(gold), cg, ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lite", help="'lite', 'verified' or a tasks .jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--base-urls", default="http://127.0.0.1:8080", help="comma-separated servers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--hint", choices=("bm25", "none"), default="bm25")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sample", type=int, help="random subset of this size (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="cache/repos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    insts = load_instances(args.dataset, args.split)[args.offset:]
    if args.sample:
        insts = sorted(random.Random(args.seed).sample(insts, min(args.sample, len(insts))),
                       key=lambda i: i["instance_id"])
    insts = insts[:args.limit] if args.limit else insts
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["instance_id"] for line in out.read_text().splitlines() if line.strip()}
    todo = [i for i in insts if i["instance_id"] not in done]
    urls = itertools.cycle(args.base_urls.split(","))
    lock = threading.Lock()
    prep_lock = threading.Lock()

    def work(inst: dict) -> dict:
        with prep_lock:
            gold, cg, ws = prepare(inst, Path(args.cache_dir))
        with lock:
            url = next(urls)
        llm = OpenAIChat(url, max_tokens=args.max_tokens, thinking=args.thinking)
        tools = file_tools(ws) + (graph_tools(cg) if args.condition == "graph" else [])
        hint = ""
        if args.hint == "bm25":
            top = rank_files(inst["problem_statement"], ws.sources)[:10]
            hint = "Files that lexically match the issue (may be wrong):\n" + "\n".join(top)
        ep = run_episode(inst["problem_statement"], inst["repo"], tools, llm, budget=args.budget, hint=hint)
        ents, files = normalize(ep["final"], cg)
        py_gold_files = [f for f in gold["files"] if f.endswith(".py")]
        return {
            "instance_id": inst["instance_id"], "repo": inst["repo"], "condition": args.condition,
            "gold": gold, "raw_final": ep["final"], "pred_entities": ents, "pred_files": files,
            "entity": rank_metrics(ents, gold["entities"]), "file": rank_metrics(files, py_gold_files),
            **{k: ep[k] for k in ("tool_calls", "invalid", "secs", "prompt_tokens", "completion_tokens", "llm_secs")},
            "steps": ep["steps"], "messages": ep["messages"],
        }

    t0 = time.time()
    with out.open("a") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            inst = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"[{n}/{len(todo)}] {inst['instance_id']} FAILED {e!r}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            ent = row["entity"]["acc@5"] if row["entity"] else None
            print(f"[{n}/{len(todo)}] {row['instance_id']} ent@5={ent} calls={row['tool_calls']} "
                  f"{row['secs']}s (elapsed {time.time() - t0:.0f}s)", flush=True)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    summary = {"condition": args.condition, "n": len(rows),
               "entity": aggregate(r["entity"] for r in rows), "file": aggregate(r["file"] for r in rows),
               "mean_tool_calls": sum(r["tool_calls"] for r in rows) / max(len(rows), 1),
               "mean_secs": sum(r["secs"] for r in rows) / max(len(rows), 1),
               "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / max(len(rows), 1)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
