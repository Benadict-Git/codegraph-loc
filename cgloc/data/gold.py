from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from unidiff import PatchSet

from codegraph.parse import Definition, parse_source


@dataclass
class Gold:
    files: list[str]
    entities: list[str]
    module_level: list[str] = field(default_factory=list)
    added_files: list[str] = field(default_factory=list)


@dataclass
class _Edit:
    line: int
    indent: int | None


def _indent(text: str) -> int | None:
    stripped = text.lstrip(" \t")
    if not stripped.strip():
        return None
    return len(text) - len(stripped)


def _edits(pf) -> list[_Edit]:
    """Removed lines anchor to themselves. Pure insertions anchor to the preceding original line and
    carry their indentation; additions that directly replace removed lines are covered by the removals."""
    edits = []
    for hunk in pf:
        last_src = hunk.source_start - 1
        prev_removed = False
        block: list[str] = []
        for line in hunk:
            if line.is_added:
                if not prev_removed:
                    block.append(line.value)
                continue
            if block:
                indents = [i for i in map(_indent, block) if i is not None]
                edits.append(_Edit(last_src, min(indents, default=None)))
                block = []
            last_src = line.source_line_no
            prev_removed = line.is_removed
            if line.is_removed:
                edits.append(_Edit(line.source_line_no, None))
        if block:
            indents = [i for i in map(_indent, block) if i is not None]
            edits.append(_Edit(last_src, min(indents, default=None)))
    return edits


def _owner(defs: list[Definition], edit: _Edit) -> Definition | None:
    """Deepest non-nested def containing the edit; inserted code must also be indented inside it."""
    best = None
    for d in defs:
        if not (d.start <= edit.line <= d.end) or d.in_function:
            continue
        if edit.indent is not None and edit.indent <= d.col:
            continue
        if best is None or d.start >= best.start:
            best = d
    return best


def gold_locations(patch: str, get_source: Callable[[str], bytes | None]) -> Gold:
    files, entities, module_level, added = [], [], [], []
    for pf in PatchSet(patch):
        path = pf.path
        files.append(path)
        if not path.endswith(".py"):
            continue
        if pf.is_added_file:
            added.append(path)
            continue
        src = get_source(path)
        if src is None:
            continue
        defs = parse_source(src).definitions
        for edit in _edits(pf):
            d = _owner(defs, edit)
            if d is None:
                if path not in module_level:
                    module_level.append(path)
                continue
            eid = f"{path}::{d.qualname}"
            if eid not in entities:
                entities.append(eid)
    return Gold(list(dict.fromkeys(files)), entities, module_level, added)
