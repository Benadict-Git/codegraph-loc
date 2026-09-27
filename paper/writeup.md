# CodeGraph-Loc: Laptop-Scale Code Graphs for Small-Agent Bug Localization

*An open toolkit, dataset and audit of where small local coding agents fail to find bugs, and how much code graphs help, on free hardware.*

## Abstract

Small open models that run on a single consumer GPU struggle most with the first step of software repair: *finding the code to change*. Studying this step is expensive. Graph-based localization systems rely on heavyweight pipelines, and SWE-bench evaluation needs Docker, which free notebooks lack. We release **CodeGraph-Loc**, which has two parts:

- **`codegraph`**, a pip-installable library that builds Python repository graphs (files, classes and functions; `contains`, `imports`, `calls` and `inherits` edges) with tree-sitter in about 3 s per Django snapshot on a laptop CPU. It exposes them through an agent-ready query API.
- **A dataset** of pre-built graphs plus hand-validated, **function-level gold locations** for all 300 SWE-bench Lite instances.

A BM25 baseline finds the right *file* in its top 5 for 70.0% of issues, but finds all gold *functions* for only 36.9%. The bottleneck is within-repository navigation. Graph propagation (BM25-seeded personalized PageRank, tuned on a repository-disjoint dev split) gives a small, tuning-robust gain (36.9 → 38.3 Acc@5).

Used *actively* as agent tools, graphs help more. A Gemma 4 E4B agent running on a free Kaggle T4 reaches 32.8% function-level Acc@1, against 19.7% for BM25. Graph tools add a small, consistent gain over file tools in both E4B and 12B, but the agents use the graph as a symbol index and almost never follow call edges.

We also audit the code graphs and embeddings shipped with this competition. With the retrieval algorithm held fixed, their graph is slightly weaker than ours: it contains only call edges and misses 8.9% of gold entities. Their embeddings can only be queried by an existing symbol name, and 46% of issues contain none. Everything runs on a CPU or a free Kaggle T4.

## 1. Introduction

Coding agents loop between *localize*, *edit* and *validate*. Frontier models localize with long contexts and many tool calls. A model that fits on one 16 GB T4, such as Gemma 4 E4B or 12B, cannot: every wasted call and every irrelevant file costs accuracy. Localization is therefore a natural target for making local agents competitive. It can also be measured offline, because the reference patch tells us which functions a correct fix touches.

Code graphs should help. A call or inheritance edge links an issue that mentions a public API (`QuerySet.update`) to the private helper where the defect lives, even when the helper shares no words with the issue. Existing graph-based systems rely on bespoke pipelines and container-based evaluation.

**Contributions.**

1. **`codegraph`**: a dependency-light graph builder with import-aware call resolution and agent query primitives.
2. **A function-level localization benchmark**: a graph and gold entities for every SWE-bench Lite instance.
3. **Baselines**: BM25 and a training-free graph re-ranker tuned on a disjoint dev split.
4. **An audit of the competition's provided graph tools**, swapping only the graph.
5. **Gemma 4 agents on a free T4**, comparing file tools with graph tools across two model sizes.

## 2. Related Work

**Benchmarks.** SWE-bench (Jimenez et al., 2024) collects real GitHub issues from 12 Python repositories. SWE-bench Lite is a 300-instance subset, and SWE-bench Verified (OpenAI, 2024) is a human-validated subset. We score *localization* instead of test execution, which removes the need for containers.

**Agents and pipelines.** SWE-agent (Yang et al., 2024) designs agent-computer interfaces. AutoCodeRover (Zhang et al., 2024) adds structure-aware code search. Agentless (Xia et al., 2024) shows that a fixed file → function → line localization pipeline rivals agent loops, which underlines how central localization is.

**Code graphs for LLMs.** CodexGraph (Liu et al., 2024) queries a graph database. RepoGraph (Ouyang et al., 2025) adds line-level repository graphs to SWE agents. LocAgent (Chen et al., 2025) trains graph-guided localization agents. We are complementary: we offer a *laptop-scale* builder and a reusable per-instance dataset so that this line of work can be studied on consumer hardware.

**Graph retrieval and training data.** Personalized PageRank (Haveliwala, 2002), as used by HippoRAG (Gutiérrez et al., 2024), propagates relevance from seeds; we seed it with BM25 (Robertson & Zaragoza, 2009). SWE-Gym (Pan et al., 2024) offers executable training environments; our gold locations give execution-free supervision.

## 3. The CodeGraph-Loc Resource

### 3.1 Graph construction

`codegraph` parses every `.py` file with tree-sitter in one iterative pass. It records definitions with exact spans (decorators included), class bases, imports and call sites (callee name plus receiver). Per-file results are path-independent and cached by content hash, so snapshots of the same repository share work. A resolution pass then links names:

- **Modules** are rooted at the top of each file's package chain, which handles `src/` layouts.
- **Imports** (absolute, relative, aliased, star) become `imports` edges and name bindings. Lookups follow re-export chains (e.g. `from pkg import Model` resolved through `pkg/__init__.py`).
- **Calls** resolve in priority order: local definitions, then imported or module-level names, then `self`/`cls` methods (walking resolved base classes), then attributes of imported modules or classes, and finally a conservative global-name fallback that links only when at most 3 definitions share the name. Builtins are never linked.

Every call edge records *how* it was resolved (`local`, `import`, `self` or `name`), so users can trade precision for recall. One Django snapshot yields 2,534 files, 8,358 classes, 23,347 functions and 30,495 call edges. It builds in 2.8 s cold (0.4 s with a warm cache) and serializes to 1.4 MB.

| All 300 SWE-bench Lite snapshots | Total | Median per instance |
|---|---|---|
| Python files | 474,210 | 1,377 |
| Nodes | 7,988,266 | 35,142 |
| `contains` / `calls` / `imports` / `inherits` edges | 7.48M / 11.20M / 1.71M / 1.05M | 32,471 / 32,787 / 7,379 / 1,535 |
| Build time incl. git reads (laptop CPU) | 630 s | 1.7 s |

The dataset (graphs, gold, all results and agent trajectories) is public on Kaggle (§6).

### 3.2 Function-level gold locations

For each instance we parse the original file at `base_commit`. Each hunk of the reference patch is mapped to the deepest *non-nested* definition containing it; nested closures are attributed to the enclosing function, which an agent can name. The rules are:

- **Removed or modified lines** map to their enclosing definition.
- **Added lines that replace removed lines** inherit the removed lines' location. This covers decorator and signature edits.
- **Pure insertions** anchor to the preceding original line and are assigned to the deepest definition whose header is *less indented than the inserted code*. A statement appended to a body therefore maps to that function, a new method maps to its class, and a new top-level function is flagged as a module-level edit.

Diff hunk headers are an unreliable shortcut. In two of six hand-checked `requests` patches, the header named a different function from the one edited; our extractor was correct in all six. The rules are unit-tested on synthetic patches.

On SWE-bench Lite, every reference patch edits exactly one file. 290 instances have at least one gold entity; the other 10 are module-level only and are scored at file level.

## 4. Experiments

**Setup.** Queries are raw issue texts. Test files are excluded from the candidates; we verified that no gold file is a test file. File-level BM25 indexes path tokens plus identifier-split contents. Entity-level BM25 ranks non-nested classes and functions within the top 10 files. **Acc@k** means *all* gold items are in the top k.

### 4.1 Lexical baseline

| SWE-bench Lite test | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR |
|---|---|---|---|---|---|
| File (n=300) | 43.3 | 62.7 | 70.0 | 79.3 | 0.558 |
| Entity (n=290) | 19.7 | 31.4 | 36.9 | 47.6 | 0.323 |

| Acc@5 by repository | n | File | Entity |
|---|---|---|---|
| django | 114 | 73 | 44 |
| sympy | 77 | 62 | 28 |
| matplotlib | 23 | 57 | 17 |
| scikit-learn | 23 | 91 | 43 |
| pytest | 17 | 53 | 25 |
| sphinx | 16 | 75 | 38 |
| six others | 30 | 77 | 47 |

The **33-point gap** between file-level and entity-level Acc@5 is the central observation. Lexical overlap usually finds the right *file*, but not the right *function* in it. Matplotlib and pytest are hardest because their issues describe user-visible behaviour (plots, CLI output) in words that rarely match the internal helpers that need fixing.

### 4.2 Graph-propagated retrieval

We run personalized PageRank from the top-k BM25 entities, weighted by score, over undirected `calls`, `inherits` and `contains` edges. Nested definitions are merged into their owners and tests are excluded. The final score is `(1−β)·bm25/max + β·ppr/max`. We select k ∈ {5, 10, 20, 50}, β ∈ {0.1 … 0.7} and restart ∈ {0.3, 0.5} on the **SWE-bench Lite dev split**. Its six repositories are disjoint from test. The selected setting is k=50, β=0.5, restart=0.5, which raised dev Acc@5 from 50.0 to 59.1 (n=22).

| Entity level, Lite test (n=290) | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| BM25 | **19.7** | 31.4 | 36.9 | 47.6 | 50.3 | **0.323** |
| BM25 + PPR (dev-selected) | 17.9 | **32.4** | **38.3** | **49.0** | **52.0** | 0.317 |
| *Best test grid point (oracle, invalid)* | *18.3* | *32.1* | *40.0* | *47.9* | *50.9* | *0.318* |

Propagation trades −1.8 at rank 1 for +1.0 to +1.8 at ranks 3–10. It surfaces callers, callees and siblings of lexical hits, which helps when the defect sits one hop from the issue's vocabulary. The gain is small but robust: **36 of 40 grid settings beat BM25** on test Acc@5 (range 36.9–40.0). A static graph used *passively* as a re-ranking prior helps only modestly. We hypothesise that its larger value lies in letting an agent *actively* follow edges (§4.4).

### 4.3 Auditing the competition's provided graphs and embeddings

The competition ships, for each of its 129 public dev tasks (fastapi, rich, requests, httpx), a NetworkX code graph and 256-d node embeddings. These back the harness tools `get_code_neighbors`, `get_code_subgraph` and `search_similar_code`. We consume them locally and do not redistribute them.

**Protocol.** Gold entities come from our extractor (121 tasks have at least one). Provided node ids are mapped to ours by longest dotted-suffix match, and PPR reuses the Lite-dev configuration unchanged. We compare BM25; BM25+PPR over `codegraph`; the *same* BM25+PPR over the **provided** graph; and the provided embeddings used as the harness allows. The sandbox has no text encoder, so `search_similar_code` must resolve its query to an existing node name. We emulate an agent that feeds each identifier-like token of the issue through the harness's name resolution and interleaves the top-10 neighbours.

| Entity level, competition dev (n=121) | Acc@1 | Acc@5 | Acc@10 | Recall@10 |
|---|---|---|---|---|
| BM25 | 9.1 | 27.3 | 30.6 | 38.0 |
| BM25 + PPR over `codegraph` | 9.1 | 27.3 | **33.9** | **40.0** |
| BM25 + PPR over provided graph | 8.3 | 26.4 | 33.1 | 39.6 |
| Provided embeddings via issue symbols | 8.3 | 13.2 | 14.9 | 17.7 |
| RRF(BM25, provided-embedding centroid) | 5.8 | 16.5 | 20.7 | 25.1 |

| Per-snapshot median | fastapi (64) | rich (45) | requests (11) |
|---|---|---|---|
| Provided nodes / edges | 3,837 / 2,474 | 1,970 / 5,628 | 733 / 1,828 |
| Provided edge types | `calls` only | `calls` only | `calls` only |
| Usable entity links (provided vs `codegraph`) | 739 vs 3,134 | 1,445 vs 2,213 | 328 vs 564 |
| Gold entities in provided graph | 327 / 368 | 115 / 115 | 13 / 13 |

**Findings.**

- **These tasks are harder than Lite.** Patches touch 4.26 entities on average, versus 1.21 on Lite. File-level Acc@5 is 40.3%.
- **Propagation helps at deeper cut-offs** (+3.3 Acc@10). This matches Lite in direction.
- **With the algorithm fixed, the provided graph is slightly weaker.** It has only call edges and is dominated by tests (3,252 of 4,259 nodes in one fastapi snapshot). It lacks 46 of 515 gold entities (8.9%), and in that snapshot also misses 33 non-test library functions (e.g. `fastapi/exception_handlers.py::websocket_request_validation_exception_handler`).
- **The embeddings are hard to use from issue text.** 56 of 121 issues (46%) contain no resolvable symbol, and neighbour search around mentioned symbols trails BM25 by half (13.2 vs 27.3 Acc@5). The practical advice for agents is to call `search_similar_code` only *after* a lexical or graph step has found a relevant symbol.

### 4.4 Graph tools for small Gemma 4 agents on a free T4

**Setup.** We serve Gemma 4 E4B and 12B (instruction-tuned, 4-bit QAT GGUF) with llama.cpp on Kaggle's free 2×T4: one server per GPU, 4 parallel slots, greedy decoding, thinking off. Measured E4B throughput is 50 tok/s single-stream and 153 tok/s with 4 parallel requests. The agent is a JSON tool-calling loop with a 12-call budget. Both conditions share the same prompt and the same top-10 BM25 file hint:

- **(A) files:** `grep`, `list_dir`, `read_file`.
- **(B) graph:** (A) plus `codegraph`'s `search`, `outline`, `read_entity`, `callers` and `callees`.

The agent returns up to 5 `path::Qualname` answers. We map them to entities identically in both conditions: exact id, then line reference, then name within the file, then enclosing class. We report the agent's own list and a **+fill** variant that appends BM25's entity ranking after the agent's picks.

E4B covers all 300 Lite test instances. 12B covers 144 instances. A first 12B run lost one of its two servers to host-RAM exhaustion, and these are the instances the surviving server completed. Their repository mix mirrors the full set, and both 12B conditions use exactly these instances.

| Lite test (entity n) | Acc@1 | Acc@5 +fill | Acc@10 +fill | File Acc@1 | Unanswered | Calls | Time/issue |
|---|---|---|---|---|---|---|---|
| BM25 (290) | 19.7 | 36.9 | 47.6 | 43.3 | – | – | – |
| E4B, files (290) | 29.0 | 49.0 | 56.9 | 63.0 | 39 | 6.6 | 53 s |
| E4B, + graph (290) | **32.8** | **50.7** | **57.2** | **67.0** | **32** | 7.4 | 63 s |
| BM25 (139) | 22.3 | 41.7 | 55.4 | 46.5 | – | – | – |
| 12B, files (139) | 43.9 | 58.3 | 69.8 | 77.1 | 14 | 6.7 | minutes* |
| 12B, + graph (139) | **47.5** | **64.7** | **73.4** | **77.8** | **4** | 7.6 | minutes* |

\*12B timings include server restarts after host-RAM kills, so they are not comparable.

**Findings** (paired bootstrap, 95% CI):

1. **A 4B-effective model on a free T4 clearly beats lexical retrieval.** E4B with graph tools improves Acc@1 over BM25 by **+13.1** [+7.6, +19.0]. With BM25 fill, Acc@5 improves by **+13.8** [+10.0, +17.9]. File-level Acc@1 rises from 43.3 to 67.0, at about one minute per issue.
2. **Model size matters most.** On the shared 139 instances, 12B beats E4B by **+10.1 Acc@1** in both conditions (files [+2.9, +17.3]; graph [+2.2, +18.0]).
3. **Graph tools help consistently but modestly.** They add +3.8 (E4B) and +3.6 (12B) Acc@1, and +1.7 and +6.5 Acc@5 with fill. None of these is individually significant (smallest p = 0.06, for 12B Acc@5). Graph agents also leave fewer issues unanswered (32 vs 39; 4 vs 14) and emit fewer malformed replies.
4. **Agents use the graph as a symbol index, not as a graph.** Across 444 graph-condition episodes, `callers` was called 9 times and `callees` never. The gain comes from `search`, `read_entity` and `outline`, which replace grep-then-read sequences. Together with §4.2, this suggests that call edges are under-used by small models. Training or prompting explicit edge traversal is the natural next step.

## 5. Limitations

- **Static call resolution is approximate.** Dynamic dispatch and duck typing are partly missed; every edge is labelled with its resolution type so users can filter.
- **Python only for now.** Only the grammar and resolution rules are language-specific.
- **Localization is necessary, not sufficient,** for repair. Its advantage is that it can be evaluated anywhere in seconds.
- **Reference patches show one valid fix,** so Acc@k is a conservative estimate.
- **The 12B subset is not a random sample,** and two 12B servers on one T4 machine exceed host RAM. The E4B results use the full set.

## 6. Reproducibility

- **Code:** https://github.com/Benadict-Git/codegraph-loc (Apache-2.0)
- **Dataset:** https://www.kaggle.com/datasets/benadictinfanta/codegraph-loc-swebench-lite
- **Reproduction notebook** (CPU; rebuilds all Lite tables and re-runs the pipeline live): https://www.kaggle.com/code/benadictinfanta/codegraph-loc-reproduction

```bash
pip install -e ".[dev,swebench]" && pytest   # 29 tests, < 1 s
python -m cgloc.eval.run_bm25                # §4.1
python -m cgloc.data.build_dataset           # graphs + gold, ~10 min CPU
python -m cgloc.eval.run_graph_rank          # §4.2, dev selection -> test
python -m cgloc.eval.run_provided            # §4.3, needs competition data locally
python kaggle/push_agent.py <name> '<config>' # §4.4 on a Kaggle T4 (llama.cpp + Gemma 4 GGUF)
python -m cgloc.agent.rescore <runs.jsonl>   # §4.4 scoring from saved raw answers
python -m cgloc.eval.agent_report a=... b=... # §4.4 tables + bootstrap CIs
```

Only §4.4 needs a GPU.

## References

- Chen, Z. et al. (2025). LocAgent: Graph-Guided LLM Agents for Code Localization. ACL.
- Gutiérrez, B. J. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS.
- Haveliwala, T. H. (2002). Topic-Sensitive PageRank. WWW.
- Jimenez, C. E. et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR.
- Liu, X. et al. (2024). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. arXiv:2408.03910.
- OpenAI (2024). Introducing SWE-bench Verified.
- Ouyang, S. et al. (2025). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. ICLR.
- Pan, J. et al. (2024). Training Software Engineering Agents and Verifiers with SWE-Gym. arXiv:2412.21139.
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Xia, C. S. et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
- Yang, J. et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS.
- Zhang, Y. et al. (2024). AutoCodeRover: Autonomous Program Improvement. ISSTA.
