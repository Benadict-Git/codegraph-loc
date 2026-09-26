from __future__ import annotations

import re
from collections.abc import Mapping

from rank_bm25 import BM25Okapi

from codegraph.build import is_test_path
from codegraph.parse import parse_source

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    out = []
    for word in _IDENT.findall(text):
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        out.extend(p for p in parts if len(p) > 1)
        if len(parts) > 1:
            out.append(word.lower())
    return out


def _path_tokens(path: str) -> list[str]:
    return tokenize(path.replace("/", " ").replace(".py", ""))


def rank_files(query: str, sources: Mapping[str, bytes], exclude_tests: bool = True) -> list[str]:
    paths = [p for p in sorted(sources) if not (exclude_tests and is_test_path(p))]
    if not paths:
        return []
    docs = [_path_tokens(p) * 2 + tokenize(sources[p].decode("utf-8", "replace")) for p in paths]
    scores = BM25Okapi(docs).get_scores(tokenize(query))
    return [p for _, p in sorted(zip(scores, paths), key=lambda x: -x[0])]


def rank_entities(query: str, sources: Mapping[str, bytes], exclude_tests: bool = True,
                  files: list[str] | None = None) -> list[str]:
    """Rank non-nested classes/functions; `files` restricts the corpus (e.g. to top-ranked files)."""
    scores = score_entities(query, sources, exclude_tests, files)
    return sorted(scores, key=lambda i: -scores[i])


def score_entities(query: str, sources: Mapping[str, bytes], exclude_tests: bool = True,
                   files: list[str] | None = None) -> dict[str, float]:
    ids, docs = [], []
    for p in files if files is not None else sorted(sources):
        if exclude_tests and is_test_path(p):
            continue
        text = sources[p].decode("utf-8", "replace")
        lines = text.splitlines()
        for d in parse_source(sources[p]).definitions:
            if d.in_function:
                continue
            ids.append(f"{p}::{d.qualname}")
            body = "\n".join(lines[d.start - 1:d.end]) if d.kind == "function" else d.signature
            docs.append(_path_tokens(p) + tokenize(d.qualname) * 2 + tokenize(body))
    if not ids:
        return {}
    return dict(zip(ids, map(float, BM25Okapi(docs).get_scores(tokenize(query)))))
