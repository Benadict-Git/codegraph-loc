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
