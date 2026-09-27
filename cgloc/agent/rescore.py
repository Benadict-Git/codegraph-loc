"""Re-score saved agent runs from their raw answers with the current normalizer (no LLM calls)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cgloc.agent.run import normalize, prepare
from cgloc.data.swebench import load_instances
from cgloc.eval.bm25 import rank_files, score_entities
from cgloc.eval.metrics import aggregate, rank_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="agent .jsonl outputs; each is rewritten to <name>.rescored.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--cache-dir", default="cache/repos")
    args = ap.parse_args()

    insts = {i["instance_id"]: i for i in load_instances(args.dataset)}
    for path in map(Path, args.runs):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out = []
        for r in rows:
            inst = insts[r["instance_id"]]
            gold, cg, ws = prepare(inst, Path(args.cache_dir))
            ents, files = normalize(r["raw_final"], cg)
            top = rank_files(inst["problem_statement"], ws.sources)[:10]
            bm25 = score_entities(inst["problem_statement"], ws.sources, files=top)
            py_gold = [f for f in gold["files"] if f.endswith(".py")]
            r.update(pred_entities=ents, pred_files=files,
                     entity=rank_metrics(ents, gold["entities"]), file=rank_metrics(files, py_gold),
                     entity_backfill=rank_metrics(list(dict.fromkeys(ents + sorted(bm25, key=lambda i: -bm25[i]))),
                                                  gold["entities"]),
                     file_backfill=rank_metrics(list(dict.fromkeys(files + top)), py_gold))
            out.append(r)
        dest = path.with_suffix(".rescored.jsonl")
        dest.write_text("".join(json.dumps(r) + "\n" for r in out))
        summary = {k: aggregate(r[k] for r in out) for k in ("entity", "entity_backfill", "file", "file_backfill")}
        summary["empty_answers"] = sum(not r["pred_entities"] for r in out)
        dest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(dest, {k: round(v["acc@1"] * 100, 1) for k, v in summary.items() if isinstance(v, dict)},
              "empty:", summary["empty_answers"])


if __name__ == "__main__":
    main()
