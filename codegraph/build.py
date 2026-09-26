from __future__ import annotations

import argparse
import builtins
import gzip
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from .parse import Definition, FileInfo, parse_source

FORMAT = "codegraph/v1"
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".tox", ".nox", ".venv", "venv",
             ".eggs", "build", "dist", "site-packages"}
_BUILTINS = set(dir(builtins))
_MAX_NAME_CANDIDATES = 3
_MAX_DEPTH = 5


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    base = parts[-1]
    return (any(p in ("tests", "testing") for p in parts[:-1])
            or base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py")


def module_name(path: str, pkg_dirs: set[str]) -> str:
    """Import name of a file, rooted at the top of its contiguous package chain (handles src/ layouts)."""
    parts = path[:-3].split("/")
    dirs = parts[:-1]
    i = len(dirs)
    while i > 0 and "/".join(dirs[:i]) in pkg_dirs:
        i -= 1
    mod = parts[i:]
    if mod[-1] == "__init__":
        mod = mod[:-1]
    return ".".join(mod)


def node_id(path: str, qualname: str) -> str:
    return f"{path}::{qualname}"


class _Resolver:
    def __init__(self, infos: Mapping[str, FileInfo]):
        self.infos = infos
        pkg_dirs = {p.rsplit("/", 1)[0] for p in infos if p.endswith("__init__.py") and "/" in p}
        self.module_of: dict[str, str] = {}
        self.modules: dict[str, str] = {}
        for p in sorted(infos):
            mod = module_name(p, pkg_dirs)
            self.module_of[p] = mod
            if mod:
                self.modules.setdefault(mod, p)
            self.modules.setdefault(p[:-3].replace("/", ".").removesuffix(".__init__"), p)

        self.defs: dict[str, dict[str, Definition]] = {
            p: {d.qualname: d for d in info.definitions} for p, info in infos.items()}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.methods_by_name: dict[str, list[str]] = defaultdict(list)
        for p, defs in self.defs.items():
            for q, d in defs.items():
                if d.in_function:
                    continue
                if d.parent is None:
                    self.by_name[q].append(node_id(p, q))
                elif defs.get(d.parent) is not None and defs[d.parent].kind == "class":
                    self.methods_by_name[q.rsplit(".", 1)[1]].append(node_id(p, q))

        self.bindings: dict[str, dict[str, tuple]] = {}
        self.star_sources: dict[str, list[str]] = {}
        self.import_edges: set[tuple[str, str]] = set()
        for p in infos:
            self._bind_imports(p)
        self.bases: dict[str, list[str]] = {}

    def _package(self, path: str) -> list[str]:
        mod = self.module_of[path].split(".") if self.module_of[path] else []
        return mod if path.endswith("__init__.py") else mod[:-1]

    def _absolute(self, path: str, module: str, level: int) -> str:
        if level == 0:
            return module
        base = self._package(path)
        if level > 1:
            base = base[:-(level - 1)] if level - 1 <= len(base) else []
        return ".".join(base + ([module] if module else []))

    def _bind_imports(self, path: str) -> None:
        binds: dict[str, tuple] = {}
        stars: list[str] = []
        for imp in self.infos[path].imports:
            full = self._absolute(path, imp.module, imp.level)
            if not imp.names:
                target = self.modules.get(full)
                if target:
                    self.import_edges.add((path, target))
                if imp.alias:
                    if target:
                        binds[imp.alias] = ("module", target)
                else:
                    top = self.modules.get(full.split(".")[0])
                    if top:
                        binds[full.split(".")[0]] = ("module", top)
                continue
            src_file = self.modules.get(full)
            for name, alias in imp.names:
                if name == "*":
                    if src_file:
                        stars.append(src_file)
                        self.import_edges.add((path, src_file))
                    continue
                sub = self.modules.get(f"{full}.{name}" if full else name)
                if sub:
                    binds[alias] = ("module", sub)
                    self.import_edges.add((path, sub))
                elif src_file:
                    binds[alias] = ("symbol", src_file, name)
                    self.import_edges.add((path, src_file))
        self.bindings[path] = binds
        self.star_sources[path] = stars

    def symbol(self, path: str, name: str, depth: int = 0) -> str | None:
        """Resolve a top-level name visible in `path`, following re-exports and star imports."""
        if depth > _MAX_DEPTH:
            return None
        d = self.defs[path].get(name)
        if d is not None:
            return node_id(path, name)
        b = self.bindings[path].get(name)
        if b is not None:
            if b[0] == "module":
                return b[1]
            return self.symbol(b[1], b[2], depth + 1)
        for src in self.star_sources[path]:
            found = self.symbol(src, name, depth + 1)
            if found:
                return found
        return None

    def member(self, class_id: str, name: str, depth: int = 0) -> str | None:
        if depth > _MAX_DEPTH or "::" not in class_id:
            return None
        path, q = class_id.split("::", 1)
        if f"{q}.{name}" in self.defs[path]:
            return node_id(path, f"{q}.{name}")
        for base in self.bases.get(class_id, []):
            found = self.member(base, name, depth + 1)
            if found:
                return found
        return None

    def resolve_name(self, path: str, name: str, scope: str | None) -> tuple[list[str], str]:
        if scope and f"{scope}.{name}" in self.defs[path]:
            return [node_id(path, f"{scope}.{name}")], "local"
        found = self.symbol(path, name)
        if found:
            return [found], "import" if name not in self.defs[path] else "local"
        if name in _BUILTINS:
            return [], ""
        cands = self.by_name.get(name, [])
        if 0 < len(cands) <= _MAX_NAME_CANDIDATES:
            return cands, "name"
        return [], ""

    def resolve_class_base(self, path: str, base: str) -> str | None:
        head, _, rest = base.partition(".")
        if not rest:
            found, _ = self.resolve_name(path, base, None)
            return found[0] if len(found) == 1 else None
        target = self.symbol(path, head)
        if target and "::" not in target:
            return self.symbol(target, rest.split(".")[-1])
        return None


def _method_owner(defs: dict[str, Definition], d: Definition) -> str | None:
    parent = defs.get(d.parent) if d.parent else None
    return d.parent if parent is not None and parent.kind == "class" else None


def build_graph(sources: Mapping[str, bytes], cache: dict | None = None, meta: dict | None = None) -> dict:
    """Build a code graph from {relative_path: source_bytes}; `cache` maps sha1(source) -> FileInfo."""
    infos: dict[str, FileInfo] = {}
    for path, src in sources.items():
        key = hashlib.sha1(src).hexdigest()
        info = cache.get(key) if cache is not None else None
        if info is None:
            info = parse_source(src)
            if cache is not None:
                cache[key] = info
        infos[path] = info

    r = _Resolver(infos)
    nodes: list[dict] = []
    edges: set[tuple[str, str, str, str]] = set()
    for path in sorted(infos):
        info = infos[path]
        test = is_test_path(path)
        nodes.append({"id": path, "kind": "file", "name": path.rsplit("/", 1)[-1], "file": path,
                      "module": r.module_of[path], "start": 1, "end": info.n_lines, "test": test})
        for d in info.definitions:
            nid = node_id(path, d.qualname)
            nodes.append({"id": nid, "kind": d.kind, "name": d.qualname.rsplit(".", 1)[-1], "file": path,
                          "start": d.start, "end": d.end, "sig": d.signature, "test": test,
                          "nested": d.in_function})
            parent = node_id(path, d.parent) if d.parent else path
            edges.add((parent, nid, "contains", ""))
            if d.kind == "class":
                resolved = [b for b in (r.resolve_class_base(path, base) for base in d.bases) if b]
                r.bases[nid] = resolved
                edges.update((nid, b, "inherits", "") for b in resolved)

    for src, dst in r.import_edges:
        if src != dst:
            edges.add((src, dst, "imports", ""))

    for path, info in infos.items():
        defs = r.defs[path]
        for d in info.definitions:
            if not d.calls:
                continue
            nid = node_id(path, d.qualname)
            cls = _method_owner(defs, d)
            for call in d.calls:
                targets, how = _resolve_call(r, path, d, cls, call.name, call.receiver)
                edges.update((nid, t, "calls", how) for t in targets if t != nid)

    return {
        "format": FORMAT,
        "meta": dict(meta or {}),
        "nodes": nodes,
        "edges": [list(e) for e in sorted(edges)],
    }


def _resolve_call(r: _Resolver, path: str, d: Definition, cls: str | None, name: str,
                  receiver: str | None) -> tuple[list[str], str]:
    if receiver is None:
        return r.resolve_name(path, name, d.qualname)
    if receiver in ("self", "cls") and cls is not None:
        found = r.member(node_id(path, cls), name)
        if found:
            return [found], "self"
    elif receiver in r.bindings[path] or receiver in r.defs[path]:
        target = r.symbol(path, receiver)
        if target and "::" not in target:
            found = r.symbol(target, name)
            return ([found], "import") if found else ([], "")
        if target:
            found = r.member(target, name)
            return ([found], "import") if found else ([], "")
    cands = r.methods_by_name.get(name, [])
    if len(cands) == 1:
        return cands, "name"
    return [], ""


def read_python_files(root: str | Path) -> dict[str, bytes]:
    root = Path(root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for f in filenames:
            if f.endswith(".py"):
                p = Path(dirpath) / f
                if p.is_file() and not p.is_symlink():
                    out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def build_from_dir(root: str | Path, meta: dict | None = None) -> dict:
    return build_graph(read_python_files(root), meta=meta)


def save_graph(graph: dict, path: str | Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(graph, f, separators=(",", ":"))


def load_graph(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a code graph for a Python repository.")
    ap.add_argument("repo", help="path to repository root")
    ap.add_argument("-o", "--output", required=True, help="output .json.gz path")
    args = ap.parse_args()
    graph = build_from_dir(args.repo, meta={"repo": os.path.abspath(args.repo)})
    save_graph(graph, args.output)
    kinds = defaultdict(int)
    for e in graph["edges"]:
        kinds[e[2]] += 1
    print(f"{len(graph['nodes'])} nodes, {dict(kinds)} edges -> {args.output}")


if __name__ == "__main__":
    main()
