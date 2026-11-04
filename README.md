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
pytest                                          # 34 unit tests, < 1 s

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

## Benchmark Scale

Pre-built code graphs and function-level gold locations across 800 SWE-bench instances (100% evaluated on CPU / free Kaggle T4):

| Metric | SWE-bench Lite (300) | SWE-bench Verified (500) | Total Resource (800) |
|---|---|---|---|
| Python files | 474,210 | 837,380 | 1,311,590 |
| AST nodes | 7.99M | 13.18M | 21.17M |
| `contains` / `calls` / `imports` / `inherits` edges | 7.5M / 11.1M / 1.7M / 1.0M | 12.3M / 16.0M / 2.8M / 2.0M | 19.8M / 27.1M / 4.5M / 3.1M |
| Build time incl. git reads (laptop CPU) | 1,942 s (5.8 s med) | 2,765 s (5.0 s med) | 4,706 s (5.2 s med) |

## Key Findings

- **Active graphs beat file search**: Gemma 4 12B gains +6.4 points Acc@1 ($p=0.006$), Gemma 4 E4B gains +3.7 points on Verified ($p=0.035$), and +3.7 points across the pooled benchmark of 682 evaluated instances ($p=0.015$).
- **Hallucination suppression**: Grounded symbol tables reduce invented method names by 60% on 12B (10.3% $\to$ 4.1%).
- **Passive graphs do not transfer**: Seeded personalized PageRank improves Lite dev but fails to transfer to Verified (-1.4 Acc@5), showing static graphs must be actively queried by models.
- **Symbol lookup over traversal**: Small models use graphs as exact symbol indices (`search`, `read_entity`), rarely traversing edges; automated neighborhood expansion (`expanded`) clutters context without gain (+0.3 Acc@1, $p=0.945$).

## License

Apache-2.0

## Citation

```bibtex
@misc{infanta2024codegraphloc,
  title={CodeGraph-Loc: Laptop-Scale Code Graphs for Small-Agent Bug Localization},
  author={Infanta, Benadict},
  year={2024},
  publisher={Kaggle},
  howpublished={\url{https://github.com/Benadict-Git/codegraph-loc}}
}
```
