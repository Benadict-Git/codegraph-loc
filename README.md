# codegraph-loc

Laptop-scale repository code graphs and function-level gold locations for studying **graph-guided bug localization** with small, local LLM agents. It was built for the Gemma 4 Developer Agent Paper Track.

Everything below runs on a CPU. The optional agent experiments run on a free Kaggle T4.

- **Dataset** (graphs, gold locations, all results, agent trajectories): https://www.kaggle.com/datasets/benadictinfanta/codegraph-loc-swebench-lite
- **Reproduction notebook**: https://www.kaggle.com/code/benadictinfanta/codegraph-loc-reproduction

## What's inside

| Path | Contents |
|---|---|
| `codegraph/` | tree-sitter parser, graph builder (file / class / function nodes; `contains`, `imports`, `calls`, `inherits` edges with call-resolution labels) and `CodeGraph` query API (`search`, `callers`, `callees`, `outline`, `neighbors`, `read`, `locate`) |
| `cgloc/data/` | SWE-bench loading, blob-level git access (no checkouts), gold file/entity extraction from reference patches, dataset builder |
| `cgloc/eval/` | metrics, BM25 baselines, BM25-seeded personalized PageRank, audit of the competition-provided graphs/embeddings |
| `cgloc/agent/` | JSON tool-calling localization agent (file tools vs file + graph tools) for any OpenAI-compatible server (llama.cpp, vLLM) |
| `kaggle/` | Kaggle GPU job scripts (llama.cpp build + agent runs) |
| `paper/` | the writeup |

## Quick start

```bash
pip install -e ".[dev,swebench]"
pytest                                          # 29 unit tests, < 1 s

codegraph-build path/to/repo -o graph.json.gz   # graph for any Python repo
```

```python
from codegraph import CodeGraph
cg = CodeGraph.load("graph.json.gz", repo_root="path/to/repo")
cg.search("QuerySet.update")
cg.callers("django/db/models/query.py::QuerySet.update")
```

## Reproducing the paper

```bash
python -m cgloc.eval.run_bm25                 # BM25 file/entity localization, SWE-bench Lite test
python -m cgloc.data.build_dataset            # graphs + gold for all 300 Lite instances (~10 min CPU)
python -m cgloc.eval.run_graph_rank           # PPR re-ranking: select on Lite dev, report on test
```

Repositories are bare-cloned into `cache/repos/` and read directly from git objects at each base commit.

**Competition audit.** The competition data is not redistributed here. Accept the competition rules, download `tasks.jsonl`, `graphs/` and `embeddings/` into `cache/kaggle/main/`, then run:

```bash
python -m cgloc.eval.run_provided --config top_k=50,beta=0.5,restart=0.5
```

**Agent runs.** Start a llama.cpp server with a Gemma 4 GGUF, then run:

```bash
python -m cgloc.agent.run --condition files --base-urls http://127.0.0.1:8080 --out outputs/files.jsonl
python -m cgloc.agent.run --condition graph --base-urls http://127.0.0.1:8080 --out outputs/graph.jsonl
python -m cgloc.eval.agent_report files=outputs/files.jsonl graph=outputs/graph.jsonl
```

## License

Apache-2.0
