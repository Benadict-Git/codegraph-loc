from __future__ import annotations

import json
from pathlib import Path

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}
FIELDS = ("instance_id", "repo", "base_commit", "patch", "problem_statement", "created_at")


def load_instances(name: str = "lite", split: str = "test") -> list[dict]:
    """Load SWE-bench instances from a HF dataset alias/name or a local .jsonl file."""
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in Path(name).read_text().splitlines() if line.strip()]
    from datasets import load_dataset

    ds = load_dataset(DATASETS.get(name, name), split=split)
    return [{k: row[k] for k in FIELDS if k in row} for row in ds]
