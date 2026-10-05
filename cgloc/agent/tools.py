from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from codegraph import CodeGraph
from codegraph.build import is_test_path

MAX_OBS_CHARS = 3000
MAX_READ_LINES = 120


@dataclass
class Tool:
    name: str
    signature: str
    description: str
    fn: Callable[..., str]


def _clip(text: str, limit: int = MAX_OBS_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


class Workspace:
    """Read-only view of one repository snapshot held in memory."""

    def __init__(self, sources: Mapping[str, bytes]):
        self.sources = sources
        self._text: dict[str, list[str]] = {}

    def lines(self, path: str) -> list[str] | None:
        if path not in self.sources:
            return None
        if path not in self._text:
            self._text[path] = self.sources[path].decode("utf-8", "replace").splitlines()
        return self._text[path]

    def text(self, path: str) -> str | None:
        lines = self.lines(path)
        return None if lines is None else "\n".join(lines)


def file_tools(ws: Workspace) -> list[Tool]:
    def grep(pattern: str, path_glob: str = "*") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"invalid regex: {e}"
        hits = []
        for path in sorted(ws.sources):
            if is_test_path(path) or not fnmatch.fnmatch(path, path_glob):
                continue
            for i, line in enumerate(ws.lines(path), 1):
                if rx.search(line):
                    hits.append(f"{path}:{i}: {line.strip()[:160]}")
                    if len(hits) >= 40:
                        return "\n".join(hits) + "\n... [more matches omitted; refine the pattern]"
        return "\n".join(hits) or "no matches"

    def list_dir(path: str = "") -> str:
        prefix = path.strip("/") + "/" if path.strip("/") else ""
        entries = set()
        for p in ws.sources:
            if p.startswith(prefix):
                rest = p[len(prefix):]
                entries.add(rest.split("/", 1)[0] + ("/" if "/" in rest else ""))
        return "\n".join(sorted(entries)[:200]) or f"no such directory: {path}"

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        lines = ws.lines(path)
        if lines is None:
            return f"no such file: {path}"
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        body = "\n".join(f"{i:>5} {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n... [file has {len(lines)} lines]" if end < len(lines) else ""
        return body + more

    return [
        Tool("grep", "grep(pattern, path_glob='*')",
             "Regex search over non-test .py files; returns path:line: text (max 40 hits).", grep),
        Tool("list_dir", "list_dir(path='')", "List files and sub-directories under a directory.", list_dir),
        Tool("read_file", "read_file(path, start_line=1, end_line=None)",
             f"Read up to {MAX_READ_LINES} numbered lines of a file.", read_file),
    ]


def _fmt(cg: CodeGraph, ids: list[str], limit: int = 25) -> str:
    rows = []
    for i in ids[:limit]:
        n = cg.get(i)
        if n is not None:
            rows.append(f"{i}  [{n['kind']}, lines {n['start']}-{n['end']}]")
    more = f"\n... [{len(ids) - limit} more]" if len(ids) > limit else ""
    return "\n".join(rows) + more if rows else "none"


def graph_tools(cg: CodeGraph) -> list[Tool]:
    def resolve(node: str) -> str | None:
        if cg.get(node) is not None:
            return node
        hits = cg.search(node, limit=1)
        return hits[0]["id"] if hits else None

    def search(query: str) -> str:
        hits = cg.search(query, limit=15)
        return "\n".join(f"{h['id']}  [{h['kind']}] {h.get('sig', '')[:100]}" for h in hits) or "no matches"

    def outline(path: str) -> str:
        if cg.get(path) is None:
            return f"no such file: {path}"
        return "\n".join(f"{'  ' * lvl}{n['id'].split('::', 1)[1]}  [{n['kind']}, lines {n['start']}-{n['end']}]"
                         for lvl, n in cg.outline(path)) or "no definitions"

    def read_entity(node: str) -> str:
        nid = resolve(node)
        return (f"{nid}\n" + (cg.read(nid, max_lines=MAX_READ_LINES) or "")) if nid else f"not found: {node}"

    def callers(node: str) -> str:
        nid = resolve(node)
        return f"callers of {nid}:\n" + _fmt(cg, [c for c in cg.callers(nid) if not cg.get(c)["test"]]) \
            if nid else f"not found: {node}"

    def callees(node: str) -> str:
        nid = resolve(node)
        return f"callees of {nid}:\n" + _fmt(cg, cg.callees(nid)) if nid else f"not found: {node}"

    return [
        Tool("search", "search(query)",
             "Find classes/functions/files whose name or path matches `query` (e.g. 'QuerySet.update').", search),
        Tool("outline", "outline(path)", "List the classes, methods and functions defined in a file.", outline),
        Tool("read_entity", "read_entity(node)",
             "Show the source of a class/function by id `path::Qualname` (or a name to look up).", read_entity),
        Tool("callers", "callers(node)", "Functions that call `node` (static call graph, tests excluded).", callers),
        Tool("callees", "callees(node)", "Functions called by `node` (static call graph).", callees),
    ]


def call_tool(tools: dict[str, Tool], name: str, args: dict) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"unknown tool '{name}'. Available: {', '.join(tools)}"
    try:
        return _clip(str(tool.fn(**args)))
    except TypeError as e:
        return f"bad arguments for {tool.signature}: {e}"
    except Exception as e:
        return f"tool error: {e!r}"


def expanded_graph_tools(cg: CodeGraph, k: int = 10) -> list[Tool]:
    """Graph tools whose `search`/`read_entity` observations always show the entity's callers and callees."""
    tools = {t.name: t for t in graph_tools(cg)}

    def edges(nid: str) -> str:
        callers = [c for c in cg.callers(nid) if not cg.get(c)["test"]][:k]
        return f"called by: {', '.join(callers) or 'none'}\ncalls: {', '.join(cg.callees(nid)[:k]) or 'none'}"

    def read_entity(node: str) -> str:
        nid = node if cg.get(node) is not None else next((h["id"] for h in cg.search(node, limit=1)), None)
        if nid is None:
            return f"not found: {node}"
        return f"{nid}\n{edges(nid)}\n" + (cg.read(nid, max_lines=MAX_READ_LINES) or "")

    def search(query: str) -> str:
        hits = cg.search(query, limit=15)
        if not hits:
            return "no matches"
        top = next((h["id"] for h in hits if h["kind"] != "file"), None)
        head = f"top hit {top}\n{edges(top)}\n\n" if top else ""
        return head + "\n".join(f"{h['id']}  [{h['kind']}] {h.get('sig', '')[:100]}" for h in hits)

    tools["read_entity"] = Tool("read_entity", "read_entity(node)",
                                "Show the source of a class/function by id `path::Qualname` (or a name to look up), "
                                "with the functions that call it and that it calls.", read_entity)
    tools["search"] = Tool("search", "search(query)",
                           "Find classes/functions/files whose name or path matches `query`; the top hit also lists "
                           "its callers and callees.", search)
    return list(tools.values())


def provided_tools(graph: dict, embeddings: Mapping[str, object] | None) -> list[Tool]:
    """Emulation of the competition harness tools over its provided NetworkX graph and node embeddings."""
    import json
    from collections import defaultdict

    import numpy as np

    from cgloc.eval.run_provided import resolve_symbol

    texts = {n["id"]: n.get("text", "") for n in graph["nodes"]}
    out_e, in_e = defaultdict(list), defaultdict(list)
    for e in graph.get("edges", graph.get("links", [])):
        out_e[e["source"]].append((e["target"], str(e.get("type", ""))))
        in_e[e["target"]].append((e["source"], str(e.get("type", ""))))

    def index(keys: list[str]) -> dict:
        lower: dict = defaultdict(list)
        for key in keys:
            lower[key.lower()].append(key)
        lower["__exact__"] = set(keys)
        return lower

    node_keys = list(texts)
    node_index = index(node_keys)
    emb_keys = [key for key in (embeddings.keys() if embeddings is not None else []) if key in texts]
    emb_index = index(emb_keys)
    mat = np.stack([np.asarray(embeddings[key], dtype=np.float32) for key in emb_keys]) if emb_keys else None
    if mat is not None:
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)

    def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        nid = resolve_symbol(node, node_keys, node_index)
        if nid is None:
            return json.dumps({"status": "error", "message": f"node not found: {node}"})
        want = str(edge_type).lower() if edge_type else None
        nbrs = list(dict.fromkeys(t for t, typ in out_e[nid] + in_e[nid] if want is None or typ.lower() == want))
        nbrs = nbrs[:int(max_neighbors)]
        return json.dumps({"status": "ok", "node": nid, "neighbors": nbrs, "count": len(nbrs)})

    def search_similar_code(query: str, k: int = 10) -> str:
        key = resolve_symbol(query, emb_keys, emb_index) if mat is not None else None
        if key is None:
            return json.dumps({"status": "error", "message": f"no embedding matches '{query}'; pass a symbol name"})
        i = emb_keys.index(key)
        sims = mat @ mat[i]
        order = [j for j in np.argsort(-sims) if j != i][:int(k)]
        return json.dumps({"status": "ok", "query": key, "count": len(order), "results": [
            {"node_name": emb_keys[j], "code": texts[emb_keys[j]][:250], "similarity": round(float(sims[j]), 4)}
            for j in order]})

    def get_code_subgraph(nodes: list[str]) -> str:
        ids = list(dict.fromkeys(n for n in (resolve_symbol(str(x), node_keys, node_index) for x in nodes) if n))
        keep = set(ids)
        edges = [{"from": s, "to": t, "type": typ} for s in ids for t, typ in out_e[s] if t in keep]
        return json.dumps({"status": "ok", "nodes": ids, "edges": edges, "node_count": len(ids),
                           "edge_count": len(edges)})

    return [
        Tool("get_code_neighbors", "get_code_neighbors(node, edge_type=None, max_neighbors=50)",
             "Incoming and outgoing neighbors of a symbol (dotted name, e.g. 'fastapi.routing.APIRoute') in the "
             "repository call graph.", get_code_neighbors),
        Tool("search_similar_code", "search_similar_code(query, k=10)",
             "Graph nodes most similar to an existing symbol name (not free text), by embedding cosine similarity.",
             search_similar_code),
        Tool("get_code_subgraph", "get_code_subgraph(nodes)",
             "Induced call subgraph (nodes and edges) for a list of symbol names.", get_code_subgraph),
    ]
