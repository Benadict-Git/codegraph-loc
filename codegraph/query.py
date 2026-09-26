from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

from .build import load_graph

SourceFn = Callable[[str], "str | None"]


def dir_source(root: str | Path) -> SourceFn:
    root = Path(root)

    def read(path: str) -> str | None:
        p = root / path
        return p.read_text("utf-8", errors="replace") if p.is_file() else None

    return read


class CodeGraph:
    def __init__(self, data: dict, source: SourceFn | None = None):
        self.meta = data.get("meta", {})
        self.nodes: dict[str, dict] = {n["id"]: n for n in data["nodes"]}
        self.out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.inc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for src, dst, kind, _ in data["edges"]:
            self.out[src].append((dst, kind))
            self.inc[dst].append((src, kind))
        self._source = source
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path: str | Path, repo_root: str | Path | None = None) -> CodeGraph:
        return cls(load_graph(path), dir_source(repo_root) if repo_root else None)

    def get(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    def search(self, query: str, kind: str | None = None, limit: int = 10,
               include_tests: bool = False) -> list[dict]:
        q = query.strip()
        ql = q.lower()
        scored = []
        for n in self.nodes.values():
            if (kind and n["kind"] != kind) or (n["test"] and not include_tests):
                continue
            nid, name = n["id"], n["name"]
            qual = nid.split("::", 1)[1] if "::" in nid else nid
            if name == q or qual == q:
                s = 100
            elif name.lower() == ql or qual.lower() == ql:
                s = 90
            elif qual.endswith("." + q) or nid.endswith(q):
                s = 85
            elif ql in name.lower():
                s = 60
            elif ql in nid.lower():
                s = 40
            else:
                continue
            if n.get("nested"):
                s -= 5
            scored.append((-s, len(nid), nid))
        scored.sort()
        return [self.nodes[nid] for _, _, nid in scored[:limit]]

    def _edges(self, node_id: str, kind: str, incoming: bool) -> list[str]:
        edges = self.inc if incoming else self.out
        return sorted({t for t, k in edges.get(node_id, []) if k == kind})

    def callers(self, node_id: str) -> list[str]:
        return self._edges(node_id, "calls", incoming=True)

    def callees(self, node_id: str) -> list[str]:
        return self._edges(node_id, "calls", incoming=False)

    def children(self, node_id: str) -> list[str]:
        return sorted(self._edges(node_id, "contains", incoming=False),
                      key=lambda i: self.nodes[i]["start"])

    def parent(self, node_id: str) -> str | None:
        parents = self._edges(node_id, "contains", incoming=True)
        return parents[0] if parents else None

    def bases(self, node_id: str) -> list[str]:
        return self._edges(node_id, "inherits", incoming=False)

    def subclasses(self, node_id: str) -> list[str]:
        return self._edges(node_id, "inherits", incoming=True)

    def imports(self, file_id: str) -> list[str]:
        return self._edges(file_id, "imports", incoming=False)

    def imported_by(self, file_id: str) -> list[str]:
        return self._edges(file_id, "imports", incoming=True)

    def neighbors(self, node_id: str, hops: int = 1, kinds: tuple[str, ...] | None = None,
                  limit: int = 50) -> list[tuple[str, int]]:
        seen = {node_id: 0}
        queue = deque([node_id])
        while queue and len(seen) <= limit:
            cur = queue.popleft()
            if seen[cur] >= hops:
                continue
            for nxt, k in self.out.get(cur, []) + self.inc.get(cur, []):
                if (kinds is None or k in kinds) and nxt not in seen:
                    seen[nxt] = seen[cur] + 1
                    queue.append(nxt)
        return [(n, d) for n, d in seen.items() if n != node_id][:limit]

    def outline(self, file_id: str, depth: int = 2) -> list[tuple[int, dict]]:
        out = []

        def walk(nid: str, level: int) -> None:
            for c in self.children(nid):
                out.append((level, self.nodes[c]))
                if level + 1 < depth and not self.nodes[c].get("nested"):
                    walk(c, level + 1)

        walk(file_id, 0)
        return out

    def lines(self, path: str) -> list[str] | None:
        if path not in self._cache:
            if self._source is None:
                return None
            text = self._source(path)
            if text is None:
                return None
            self._cache[path] = text.splitlines()
        return self._cache[path]

    def read(self, node_id: str, max_lines: int = 200) -> str | None:
        n = self.nodes.get(node_id)
        if n is None:
            return None
        lines = self.lines(n["file"])
        if lines is None:
            return None
        start, end = n["start"], min(n["end"], n["start"] + max_lines - 1)
        body = "\n".join(f"{i:>5} {lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))
        if end < n["end"]:
            body += f"\n... ({n['end'] - end} more lines)"
        return body

    def locate(self, path: str, line: int) -> str | None:
        """Innermost definition containing `line` in `path`, or the file id."""
        best = path if path in self.nodes else None
        for c in self._walk_defs(path):
            n = self.nodes[c]
            if n["start"] <= line <= n["end"]:
                best = c
        return best

    def _walk_defs(self, nid: str):
        for c in self.children(nid):
            yield c
            yield from self._walk_defs(c)
