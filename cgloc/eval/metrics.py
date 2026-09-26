from __future__ import annotations

from collections.abc import Iterable, Sequence

KS = (1, 3, 5, 10)


def rank_metrics(pred: Sequence[str], gold: Iterable[str], ks: Sequence[int] = KS) -> dict[str, float] | None:
    """acc@k: all gold items in top-k; recall@k: fraction of gold in top-k; mrr: first gold hit."""
    gold = set(gold)
    if not gold:
        return None
    pred = list(dict.fromkeys(pred))
    out = {}
    for k in ks:
        top = set(pred[:k])
        out[f"acc@{k}"] = float(gold <= top)
        out[f"recall@{k}"] = len(gold & top) / len(gold)
    out["mrr"] = next((1.0 / (i + 1) for i, p in enumerate(pred) if p in gold), 0.0)
    return out


def aggregate(rows: Iterable[dict[str, float] | None]) -> dict[str, float]:
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"n": 0}
    out = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    out["n"] = len(rows)
    return out
