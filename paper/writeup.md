# CodeGraph-Loc: Laptop-Scale Code Graphs and Function-Level Gold Locations for Graph-Guided Bug Localization

**Subtitle:** An open toolkit and dataset that lets anyone study how repository graphs help small, local LLM agents find where a bug lives, on free hardware and without Docker.

---

## Abstract

Small open models that run on a single consumer accelerator struggle most with the first step of software repair: *finding the code to change*. Research on this step is held back by infrastructure. Graph-based localization systems depend on heavyweight code-graph pipelines, and end-to-end SWE-bench evaluation needs containerized execution, which is unavailable on free notebooks. We release **CodeGraph-Loc**, which has two parts. The first is `codegraph`, a small pip-installable library that builds Python repository graphs (files, classes and functions; `contains`, `imports`, `calls` and `inherits` edges) with tree-sitter in about 3 s per Django snapshot on a laptop CPU, and exposes them through an agent-ready query API. The second is a dataset of pre-built graphs plus **function-level gold locations** for every SWE-bench Lite instance, extracted from the reference patches with an indentation-aware insertion rule that we validate by hand. On this benchmark, a BM25 baseline finds the right file in its top 5 for 70.0% of issues, but finds all gold functions in its top 5 for only 36.9%. Most of the difficulty for small models is therefore *within-repository navigation*, which graphs are designed to address. Using the graph passively, by seeding personalized PageRank with BM25 scores, gives a small but tuning-robust gain (Acc@5 36.9 → 38.3, with configurations chosen on a repository-disjoint dev split). We also audit the code graphs and embeddings shipped with the Gemma 4 Developer Agent competition. With the retrieval algorithm held fixed, their graph is slightly weaker than ours; it lacks 8.9% of the gold entities and has only call edges. Their embeddings can only be queried by an existing symbol name, which makes them hard to use from issue text: 46% of issues contain no resolvable symbol. All code, data and evaluation scripts run on a CPU or a free Kaggle T4.

## 1. Introduction

Autonomous coding agents typically loop between *localize* (which files and functions are relevant?), *edit* and *validate*. For frontier API models, localization is often solved implicitly by long contexts and many tool calls. For models that fit on one consumer GPU, such as the Gemma 4 family on a 16 GB T4, every wasted tool call and every irrelevant file in context directly costs accuracy. Localization is therefore a natural target for making local agents competitive. It is also measurable offline: given the reference patch, we know which functions a correct fix touches.

Code graphs are a promising tool here. A call or inheritance edge can connect an issue that mentions a public API (`QuerySet.update`) to the private helper where the defect actually lives, even when the helper shares no words with the issue text. Recent systems exploit this, but they rely on bespoke pipelines that are costly to reproduce. Their evaluation also usually requires Docker-based test execution, which excludes most free and consumer compute.

We contribute:

1. **`codegraph`**, a dependency-light (tree-sitter + stdlib) library that builds a typed repository graph with import-aware call resolution (re-exports, relative imports, `src/` layouts, `self.method()` resolved through base classes). It serializes to compact gzipped JSON and exposes `search`, `callers`, `callees`, `outline`, `neighbors`, `read` and `locate`, the primitives an agent tool layer needs.
2. **A function-level localization benchmark** built on SWE-bench Lite: one graph per instance at its base commit, plus gold files, gold entities (non-nested functions, methods or classes) and module-level edit flags. Extraction is deterministic and validated by hand.
3. **Baselines and analysis**: lexical retrieval (BM25) at file and entity level, a per-repository breakdown, and a training-free graph method (BM25-seeded personalized PageRank) whose hyper-parameters are chosen on the disjoint SWE-bench Lite *dev* split.
4. **A fully reproducible pipeline**: every number in this paper is produced by one command on a CPU. Blob-level parse caching makes building all 300 graphs a matter of minutes.

## 2. Related Work

**Benchmarks.** SWE-bench (Jimenez et al., 2024) collects real GitHub issues from 12 Python repositories, with execution-based evaluation. SWE-bench Lite is a 300-instance subset of self-contained issues, and SWE-bench Verified (OpenAI, 2024) is a human-validated subset. We reuse their instances but evaluate *localization* rather than test execution, which removes the need for containers.

**Agents and pipelines.** SWE-agent (Yang et al., 2024) designs agent-computer interfaces for repository navigation. AutoCodeRover (Zhang et al., 2024) exposes structure-aware code search APIs (class and method lookup). Agentless (Xia et al., 2024) shows that a fixed hierarchical pipeline (file, then class/function, then edit location) is competitive with agent loops, which highlights how central localization is.

**Code graphs for LLMs.** CodexGraph (Liu et al., 2024) stores repositories in a graph database queried by the LLM. RepoGraph (Ouyang et al., 2025) builds line-level repository graphs as a plug-in for SWE agents. LocAgent (Chen et al., 2025) builds a heterogeneous code graph and trains graph-guided localization agents. Our work is complementary. We focus on a *lightweight, laptop-scale* graph builder and a reusable per-instance dataset, so that graph-guided localization can be studied under consumer-hardware constraints.

**Graph-based retrieval.** Personalized PageRank (Haveliwala, 2002) propagates relevance from seed nodes. HippoRAG (Gutiérrez et al., 2024) uses it over knowledge graphs for multi-hop retrieval. We apply the same idea to code entities, seeding with BM25 (Robertson & Zaragoza, 2009) scores.

**Training data for SWE agents.** SWE-Gym (Pan et al., 2024) provides executable training environments. Our graphs and gold locations can supply dense, execution-free supervision for localization sub-policies.

## 3. The CodeGraph-Loc Resource

### 3.1 Graph construction

For a repository snapshot, `codegraph` parses every `.py` file with tree-sitter in a single iterative (recursion-free) pass. The pass records definitions with exact spans (decorators included), class bases, imports and call sites (callee name plus receiver identifier). Per-file parse results are path-independent, so they are cached by content hash and reused across snapshots of the same repository.

A resolution pass then links names:

- **Module names** are rooted at the top of each file's contiguous package chain, which handles `src/` layouts and test directories without `__init__.py`.
- **Imports** (absolute, relative, aliased, star) become `imports` edges and per-file name bindings. Symbol lookups follow re-export chains (e.g. `from pkg import Model` resolved through `pkg/__init__.py`).
- **Calls** resolve in priority order: nested local definitions, imported or module-level names, `self`/`cls` methods (walking resolved base classes), attributes of imported modules or classes, and finally a conservative global-name fallback. The fallback links only when at most 3 top-level candidates share the name (exactly 1 for arbitrary receivers). Builtins are never linked.

Every call edge records *how* it was resolved (`local`, `import`, `self` or `name`), so users can trade precision for recall.

| Node / edge type | Count (Django, one snapshot) |
|---|---|
| files / classes / functions | 2,534 / 8,358 / 23,347 |
| `contains` / `calls` / `imports` / `inherits` | 31,606 / 30,495 / 7,175 / 6,963 |
| call resolution: import / self / name / local | 12,530 / 9,528 / 6,186 / 2,251 |

Building this graph takes 2.8 s cold and 0.4 s with a warm parse cache on a laptop CPU. The serialized graph is 1.4 MB.

**Dataset scale (all 300 SWE-bench Lite test snapshots).**

| | Total | Median per instance |
|---|---|---|
| Python files | 474,210 | 1,377 |
| Nodes (files + classes + functions) | 7,988,266 | 35,142 |
| `contains` / `calls` / `imports` / `inherits` edges | 7.48M / 11.20M / 1.71M / 1.05M | 32,471 / 32,787 / 7,379 / 1,535 |
| Build time, including reading git objects (laptop CPU) | 630 s | 1.7 s (max 18.9 s) |

The complete dataset (300 gzipped graphs plus gold annotations) is 326 MB.

### 3.2 Function-level gold locations

For each instance we parse the original file at `base_commit` and map every hunk of the reference patch to the deepest *non-nested* definition that contains it. Nested closures are attributed to their enclosing function or method, which an agent can actually name. The rules are:

- A **removed or modified line** maps to its enclosing definition.
- An **added line that directly replaces removed lines** inherits their location. This handles decorator and signature edits.
- A **pure insertion** anchors to the preceding original line, and is assigned to the deepest enclosing definition whose header is *less indented than the inserted code*. A statement appended to a function body therefore maps to that function, a new method maps to its class, and a new top-level function is flagged as a module-level edit.

This rule avoids a common failure of hunk-header heuristics: diff headers show the nearest preceding `def`, which is often wrong. In two of six hand-checked `requests` patches (`psf__requests-3362`, `psf__requests-863`) the header named a different function than the one edited, and our extractor was correct in all six. We unit-test the rules on synthetic patches covering method edits, nested closures, inserted methods, appended statements, decorator edits, import edits and new files.

**Benchmark statistics (SWE-bench Lite test, 300 instances).** Every reference patch edits exactly one file. 290 instances have at least one gold entity; the remaining 10 edit only module-level code, and we report them at file level only.

## 4. Experiments

**Setup.** Queries are the raw issue texts. Test files are excluded from the candidate set; we verified that no gold file is a test file. File-level BM25 indexes path tokens (up-weighted) plus identifier-split file contents. Entity-level BM25 ranks non-nested classes and functions within the top 10 BM25 files. We report Acc@k (all gold items appear in the top k) and MRR.

### 4.1 Lexical baseline

| Level | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR |
|---|---|---|---|---|---|
| File (n=300) | 43.3 | 62.7 | 70.0 | 79.3 | 0.558 |
| Entity (n=290) | 19.7 | 31.4 | 36.9 | 47.6 | 0.323 |

Per repository (Acc@5, %):

| Repository | n | File | Entity |
|---|---|---|---|
| django | 114 | 73 | 44 |
| sympy | 77 | 62 | 28 |
| matplotlib | 23 | 57 | 17 |
| scikit-learn | 23 | 91 | 43 |
| pytest | 17 | 53 | 25 |
| sphinx | 16 | 75 | 38 |
| others (astropy, requests, pylint, xarray, seaborn, flask) | 30 | 77 | 47 |

The 33-point gap between file-level and entity-level Acc@5 is the central observation. Lexical overlap with the issue text is usually enough to find the right *file*, but not the right *function* inside it. Matplotlib and pytest are hardest: issue reports there describe user-visible behaviour (plots, CLI output) whose vocabulary rarely matches the internal helpers that need fixing.

### 4.2 Graph-propagated retrieval

We seed personalized PageRank with the top-k BM25 entities, weighted by their scores, over an undirected graph of `calls`, `inherits` and `contains` edges. Nested definitions are merged into their owners and test code is excluded. Final scores mix normalized BM25 and PPR mass: `s = (1-β)·bm25/max + β·ppr/max`. The hyper-parameters (k ∈ {5, 10, 20, 50}, β ∈ {0.1, …, 0.7}, restart ∈ {0.3, 0.5}) are selected on the 23-instance SWE-bench Lite **dev** split. Its repositories (sqlfluff, marshmallow, pvlib, pydicom, astroid, pyvista) are disjoint from the test repositories, so the selection is not tuned on test data.

The selected configuration is k=50, β=0.5, restart=0.5. On dev it raised Acc@5 from 50.0 to 59.1 (n=22).

| Entity level, SWE-bench Lite test (n=290) | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| BM25 | **19.7** | 31.4 | 36.9 | 47.6 | 39.0 | 50.3 | **0.323** |
| BM25 + PPR (dev-selected) | 17.9 | **32.4** | **38.3** | **49.0** | **40.8** | **52.0** | 0.317 |
| *Best grid point on test (oracle, not a valid result)* | *18.3* | *32.1* | *40.0* | *47.9* | *42.4* | *50.9* | *0.318* |

Graph propagation trades a small loss at rank 1 (−1.8) for gains at ranks 3–10 (+1.0 to +1.8 Acc, +1.7 to +1.8 Recall). It surfaces callers, callees and siblings of lexically matched entities, which helps when the defect sits one hop away from the vocabulary of the issue. It can also push a correct lexical top-1 down. The effect is small but not an artifact of tuning: 36 of the 40 grid configurations beat BM25 on test Acc@5, spanning 36.9 to 40.0 (median 38.1).

Our reading is that a static graph used *passively*, as a re-ranking prior, gives only a modest improvement. Its larger value may lie in letting an agent *actively* follow edges from a lexical foothold. Section 4.4 tests that hypothesis.

### 4.3 Auditing the competition's provided graphs and embeddings

The Gemma 4 Developer Agent competition ships, for each of its 129 public development tasks (fastapi, rich, requests, httpx), a NetworkX code graph and 256-d node embeddings. These back three harness tools: `get_code_neighbors`, `get_code_subgraph` and `search_similar_code`. We apply the same analysis to them. We never redistribute the competition data; our code consumes it locally.

**Protocol.** Gold locations are extracted from each task's reference patch with our extractor (121 tasks have at least one gold entity). Provided nodes are dotted symbol paths, which we map to our `path::Qualname` ids by longest dotted-suffix match. Every method is scored on the same candidate set. Graph propagation uses exactly the configuration selected on SWE-bench Lite dev (k=50, β=0.5, restart=0.5), with no tuning on these tasks. We compare four methods:

1. BM25.
2. BM25 plus PPR over `codegraph`.
3. The *same* BM25 plus PPR over the **provided** graph. The retrieval algorithm is held fixed and only the graph changes.
4. The provided embeddings, used the way the harness allows. The sandbox has no text encoder, so `search_similar_code` resolves its query to an *existing node name* and returns that node's nearest neighbours. We emulate an agent that passes each identifier-like token of the issue (CamelCase, snake_case or dotted) through the harness's 4-tier name resolution and interleaves the top-10 neighbours.

We also report a parameter-free reciprocal-rank fusion of BM25 with embedding similarity to the centroid of the top-5 BM25 entities.

| Method (entity level, n=121) | Acc@1 | Acc@5 | Acc@10 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| BM25 | 9.1 | 27.3 | 30.6 | 33.4 | 38.0 |
| BM25 + PPR over `codegraph` | 9.1 | 27.3 | **33.9** | **33.9** | **40.0** |
| BM25 + PPR over provided graph | 8.3 | 26.4 | 33.1 | 33.4 | 39.6 |
| Provided embeddings via issue symbols | 8.3 | 13.2 | 14.9 | 15.2 | 17.7 |
| RRF(BM25, provided-embedding centroid) | 5.8 | 16.5 | 20.7 | 19.2 | 25.1 |

**Structural audit.**

| Per-snapshot median | fastapi (64) | rich (45) | requests (11) |
|---|---|---|---|
| Provided nodes / edges | 3,837 / 2,474 | 1,970 / 5,628 | 733 / 1,828 |
| Provided edge types | `calls` only | `calls` only | `calls` only |
| Undirected entity–entity links usable for propagation (provided vs `codegraph`) | 739 vs 3,134 | 1,445 vs 2,213 | 328 vs 564 |
| Gold entities present in provided graph | 327 / 368 | 115 / 115 | 13 / 13 |

**Findings.**

- **The task set is much harder than SWE-bench Lite.** Reference patches touch 4.26 entities on average, compared with 1.21 on Lite. File-level BM25 Acc@5 falls to 40.3% (47.4% recall).
- **Graph propagation helps modestly and consistently at deeper cut-offs.** It gains +3.3 Acc@10 and +2.0 Recall@10 over BM25, with no change at the top ranks. This matches the Lite result in direction but is smaller.
- **Holding the algorithm fixed, the provided graph is slightly weaker than `codegraph`.** It has only `calls` edges (no containment, import or inheritance edges). It is dominated by test code (in one fastapi snapshot, 3,252 of 4,259 nodes are under `tests.`). 46 of the 515 gold entities (8.9%) are absent from it. More broadly, in the same fastapi snapshot 33 non-test library functions and 420 documentation-example functions found by `codegraph` have no provided node; `fastapi/exception_handlers.py::websocket_request_validation_exception_handler` is one of the missing library functions.
- **The provided embeddings are hard to use from an issue description.** Because queries must name an existing symbol, 56 of 121 issues (46%) contain no identifier that resolves at all. Even when they do, nearest-neighbour search around a mentioned symbol recovers the gold entity far less often than lexical search (Acc@5 13.2 vs 27.3). Fusing embeddings with BM25 also hurts. For a small agent, the practical implication is to call `search_similar_code` only *after* a lexical or graph step has produced a relevant symbol, not as a first step.

### 4.4 Ongoing: graph tools for small Gemma 4 agents

*This section reports work in progress and will be updated before the deadline.* We are running Gemma 4 E4B and 12B (4-bit GGUF, llama.cpp) on free Kaggle T4 GPUs as JSON tool-calling localization agents under identical budgets (12 tool calls). Condition (A) has only `grep`, `list_dir` and `read_file`. Condition (B) adds `codegraph`'s `search`, `outline`, `read_entity`, `callers` and `callees`. Both conditions receive the same top-10 BM25 file hint. We report localization accuracy alongside tool calls, tokens and wall-clock time per issue, and release all trajectories.

## 5. Discussion and Limitations

- **Static resolution is approximate.** Dynamic dispatch, monkey-patching and duck typing are only partially captured. We keep the global-name fallback conservative and label every edge with its resolution type so that downstream users can filter.
- **Python only for now.** Nothing in the design is Python-specific beyond the grammar and resolution rules. tree-sitter grammars exist for most languages, and the per-file/resolution split makes ports incremental.
- **Localization is not repair.** Accuracy is a proxy; a correct location is necessary but not sufficient for a correct patch. Its advantage is that it can be evaluated on any machine in seconds.
- **Gold ambiguity.** A patch shows one valid fix; other correct fixes may touch other functions. Acc@k over the reference entities is therefore a conservative estimate.

## 6. Reproducibility

```bash
pip install -e ".[dev,swebench]"
pytest                                   # 20 unit tests, CPU, < 1 s
python -m cgloc.eval.run_bm25            # Table 4.1
python -m cgloc.data.build_dataset       # graphs + gold for all 300 instances
python -m cgloc.eval.run_graph_rank      # Section 4.2 (dev selection → test)
```

Repositories are read directly from git objects (`git cat-file --batch`) at each base commit, with no checkouts, containers or GPUs. Per-instance predictions, gold locations and summaries are written as JSONL.
[[TBD: links — public GitHub repo, Kaggle Dataset, Kaggle reproduction notebook]]

## References

- Chen, Z. et al. (2025). *LocAgent: Graph-Guided LLM Agents for Code Localization.* ACL.
- Gutiérrez, B. J. et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS.
- Haveliwala, T. H. (2002). *Topic-Sensitive PageRank.* WWW.
- Jimenez, C. E. et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR.
- Liu, X. et al. (2024). *CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases.* arXiv:2408.03910.
- OpenAI (2024). *Introducing SWE-bench Verified.*
- Ouyang, S. et al. (2025). *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph.* ICLR.
- Pan, J. et al. (2024). *Training Software Engineering Agents and Verifiers with SWE-Gym.* arXiv:2412.21139.
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR.
- Xia, C. S. et al. (2024). *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv:2407.01489.
- Yang, J. et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS.
- Zhang, Y. et al. (2024). *AutoCodeRover: Autonomous Program Improvement.* ISSTA.
- tree-sitter: https://tree-sitter.github.io
