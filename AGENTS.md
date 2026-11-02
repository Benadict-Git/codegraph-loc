# AGENTS.md

Repository code graphs and function-level gold locations for studying graph-guided bug localization with small local LLM agents (built for the Gemma 4 Paper Track).

## Commands

- **Environment**: Use `.venv/bin/python` (Python >= 3.10)
- **Install**: `pip install -e ".[dev,swebench]"`
- **Test all**: `.venv/bin/pytest -q` (34 tests, < 1s)
- **Single test**: `.venv/bin/pytest tests/test_codegraph.py::test_roundtrip_and_query -q` (or `-k <pattern>`)
- **Lint / typecheck**: None configured in repo; keep code clean and verified via tests.
- **Build graph**: `codegraph-build path/to/repo -o graph.json.gz` (or `python -m codegraph.build`)
- **Run evaluations**:
  - BM25 baseline: `.venv/bin/python -m cgloc.eval.run_bm25`
  - Dataset build (SWE-bench Lite): `.venv/bin/python -m cgloc.data.build_dataset`
  - Personalized PageRank: `.venv/bin/python -m cgloc.eval.run_graph_rank`
  - Competition graph audit: `.venv/bin/python -m cgloc.eval.run_provided --config top_k=50,beta=0.5,restart=0.5`
- **Agent evaluation & scoring**:
  - Run agent: `.venv/bin/python -m cgloc.agent.run --condition <files|graph|expanded|provided> --base-urls http://127.0.0.1:8080 --out <path>`
  - Rescore raw answers: `.venv/bin/python -m cgloc.agent.rescore <path.jsonl>`
  - Comparative report: `.venv/bin/python -m cgloc.eval.agent_report files=outputs/files.jsonl graph=outputs/graph.jsonl`
- **Kaggle / Git sync**:
  - Sync code dataset: `bash kaggle/sync_code.sh "commit message"`
  - Build & push repro notebook: `.venv/bin/python kaggle/repro_notebook/build_notebook.py && kaggle kernels push -p kaggle/repro_notebook`
  - Push to GitHub: `git -c credential.helper= -c credential.helper='!gh auth git-credential' push`

## Architecture

- `codegraph/`: Core lightweight graph engine.
  - `parse.py`: Tree-sitter AST extraction of classes, functions, calls, imports, and inheritance.
  - `build.py`: Graph constructor resolving intra-repo references, imports, and call targets (with heuristics pruning builtin collisions); formats as JSON/gzipped JSON (`codegraph/v1`).
  - `query.py`: `CodeGraph` query API providing `search`, `callers`, `callees`, `outline`, `neighbors`, `read`, and `locate`.
- `cgloc/data/`: Data extraction pipeline.
  - `swebench.py`: SWE-bench instance loader.
  - `checkout.py`: Git blob access directly from bare clones in `cache/repos/` without checking out working trees.
  - `gold.py`: Extracts ground-truth modified files and functions from benchmark reference patches.
  - `build_dataset.py`: Coordinates graph creation and gold labels across benchmark splits.
- `cgloc/agent/`: Fault localization agents running against OpenAI-compatible endpoints (llama.cpp/vLLM).
  - `tools.py`: Tool definitions for `files`, `graph`, `expanded` (proactive caller/callee context), and `provided` (competition baseline emulation).
  - `loop.py` & `run.py`: Multi-turn tool execution loop and batch runner.
  - `rescore.py`: Evaluates and re-scores agent prediction files against gold locations.
- `cgloc/eval/`: Quantitative evaluation and diagnostic analyses (`metrics.py`, `bm25.py`, `graph_rank.py`, `hop_analysis.py`, `edge_audit.py`, `failure_taxonomy.py`).
- `kaggle/`: GPU execution harness for Kaggle T4x2 environments, kernel specs, and sync scripts.
- `paper/`: Contest writeups (`kaggle_writeup.md` <= 3,000 words limit, `writeup.md`).

## Conventions & Rules

- **Workspace boundary**: Work only inside `/home/benny/Projects/google-gemma/`. Never touch `~/Projects/gemma-agent-main/` or user dotfiles.
- **Data boundaries**: Never redistribute competition data (`cache/kaggle/main/`). Public datasets and outputs must strictly use SWE-bench derived data.
- **Git object access**: Code graphs are built from bare git clones via raw blob inspection (`cgloc/data/checkout.py`), not disk checkouts.
- **Scoring integrity**: Always score agents from saved raw answers via `cgloc.agent.rescore`. Report exact measured metrics without rounding up or omitting negative findings.
- **Graph versions**: Always use v2 graphs (`1363f0c`+) which avoid false edges on builtin methods and attribute chains.
- **Kaggle management**: Only touch Kaggle resources starting with `cgloc-`, `codegraph-loc-reproduction`, and `codegraph-loc-swebench-lite`. Never delete or modify other user kernels (`g4agent-*`, `umud-*`, `rsna-*`).
