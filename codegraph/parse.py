from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)

_DEF_TYPES = ("function_definition", "class_definition")


@dataclass
class Call:
    name: str
    receiver: str | None


@dataclass
class Definition:
    qualname: str
    kind: str
    start: int
    end: int
    col: int
    signature: str
    in_function: bool
    parent: str | None
    bases: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)


@dataclass
class Import:
    module: str
    level: int
    names: list[tuple[str, str]]
    alias: str | None = None


@dataclass
class FileInfo:
    definitions: list[Definition]
    imports: list[Import]
    n_lines: int
    has_error: bool


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8", "replace") if node is not None else ""


def _signature(node: Node, src: bytes) -> str:
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    sig = " ".join(src[node.start_byte:end].decode("utf-8", "replace").split())
    return sig[:200]


def _parse_import(node: Node) -> list[Import]:
    out = []
    for child in node.named_children:
        if child.type == "dotted_name":
            out.append(Import(_text(child), 0, []))
        elif child.type == "aliased_import":
            out.append(Import(_text(child.child_by_field_name("name")), 0, [],
                              _text(child.child_by_field_name("alias"))))
    return out


def _parse_import_from(node: Node) -> Import:
    mod_node = node.child_by_field_name("module_name")
    level, module = 0, _text(mod_node)
    if mod_node is not None and mod_node.type == "relative_import":
        module = ""
        for c in mod_node.named_children:
            if c.type == "import_prefix":
                level = _text(c).count(".")
            elif c.type == "dotted_name":
                module = _text(c)
    names = []
    if any(c.type == "wildcard_import" for c in node.named_children):
        names.append(("*", "*"))
    for n in node.children_by_field_name("name"):
        if n.type == "aliased_import":
            names.append((_text(n.child_by_field_name("name")), _text(n.child_by_field_name("alias"))))
        else:
            name = _text(n)
            names.append((name, name))
    return Import(module, level, names)


def _bases(node: Node) -> list[str]:
    sup = node.child_by_field_name("superclasses")
    if sup is None:
        return []
    return [_text(c) for c in sup.named_children if c.type in ("identifier", "attribute")]


def parse_source(src: bytes) -> FileInfo:
    tree = _parser.parse(src)
    defs: list[Definition] = []
    imports: list[Import] = []
    seen_calls: dict[int, set[tuple[str, str | None]]] = {}
    # (node, enclosing qualname parts, index of def owning calls, inside a function)
    stack: list[tuple[Node, tuple[str, ...], int | None, bool]] = [(tree.root_node, (), None, False)]
    while stack:
        node, scope, owner, in_fn = stack.pop()
        t = node.type
        if t in _DEF_TYPES:
            name = _text(node.child_by_field_name("name"))
            outer = node.parent if node.parent is not None and node.parent.type == "decorated_definition" else node
            kind = "class" if t == "class_definition" else "function"
            defs.append(Definition(
                qualname=".".join(scope + (name,)),
                kind=kind,
                start=outer.start_point[0] + 1,
                end=outer.end_point[0] + 1,
                col=outer.start_point[1],
                signature=_signature(node, src),
                in_function=in_fn,
                parent=".".join(scope) or None,
                bases=_bases(node) if kind == "class" else [],
            ))
            idx = len(defs) - 1
            body = node.child_by_field_name("body")
            if body is not None:
                stack.append((body, scope + (name,), idx if kind == "function" else None,
                              in_fn or kind == "function"))
            continue
        if t == "import_statement":
            imports.extend(_parse_import(node))
            continue
        if t == "import_from_statement":
            imports.append(_parse_import_from(node))
            continue
        if t == "call" and owner is not None:
            fn = node.child_by_field_name("function")
            call = None
            if fn is not None and fn.type == "identifier":
                call = (_text(fn), None)
            elif fn is not None and fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                call = (_text(fn.child_by_field_name("attribute")),
                        _text(obj) if obj is not None and obj.type == "identifier" else None)
            if call is not None and call not in seen_calls.setdefault(owner, set()):
                seen_calls[owner].add(call)
                defs[owner].calls.append(Call(*call))
        stack.extend((c, scope, owner, in_fn) for c in reversed(node.named_children))
    defs.sort(key=lambda d: (d.start, d.col))
    n_lines = src.count(b"\n") + (0 if src.endswith(b"\n") or not src else 1)
    return FileInfo(defs, imports, n_lines, tree.root_node.has_error)
