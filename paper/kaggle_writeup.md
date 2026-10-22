## Abstract

Small open models that run on a single consumer GPU struggle most with the first step of software repair: *finding the code to change*. Studying this step is expensive. Graph-based localization systems rely on heavyweight pipelines, and SWE-bench evaluation needs Docker, which free notebooks lack. We release **CodeGraph-Loc**, which has two parts:

- **`codegraph`**, a pip-installable library that builds Python repository graphs (files, classes and functions; `contains`, `imports`, `calls` and `inherits` edges) with tree-sitter in about 3 s per Django snapshot on a laptop CPU. It exposes them through an agent-ready query API.
- **A dataset** of pre-built graphs plus hand-validated, **function-level gold locations** for 300 SWE-bench Lite and 485 SWE-bench Verified instances (785 total instances).

A BM25 baseline finds the right *file* in its top 5 for 70.0% of Lite issues, but finds all gold *functions* for only 36.9%. The bottleneck is within-repository navigation. Graph propagation (BM25-seeded personalized PageRank) offers only a minor, statistically insignificant gain on Lite and fails to transfer to the Verified set, showing that a graph used *passively* as a re-ranker is not a reliable solution.

Used *actively* as agent tools, graphs help more. A Gemma 4 E4B agent running on a free Kaggle T4 reaches 32.8% function-level Acc@1 on Lite, against 19.7% for BM25. Graph tools add a consistent gain over file tools across models and datasets, while drastically reducing hallucinated names.

We also audit the code graphs and embeddings shipped with this competition. The provided graph misses 8.9% of gold entities and its retrieval tools are weaker than baseline file search, making it the weakest agent condition. Everything runs on a CPU or a free Kaggle T4.

## 1. Introduction

Coding agents loop between *localize*, *edit* and *validate*. Frontier models localize with long contexts and many tool calls. A model that fits on one 16 GB T4, such as Gemma 4 E4B or 12B, cannot: every wasted call and every irrelevant file costs accuracy. Localization is therefore a natural target for making local agents competitive. It can also be measured offline, because the reference patch tells us which functions a correct fix touches.

Code graphs should help. A call or inheritance edge links an issue that mentions a public API (`QuerySet.update`) to the private helper where the defect lives, even when the helper shares no words with the issue. Existing graph-based systems rely on bespoke pipelines and container-based evaluation.

**Contributions.**

1. **`codegraph`**: a dependency-light graph builder with import-aware call resolution and agent query primitives.
2. **A function-level localization benchmark**: a graph and gold entities for 785 SWE-bench Lite and Verified instances.
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
- **Imports** (absolute, relative, aliased, star) become `imports` edges and name bindings. Lookups follow re-export chains.
- **Calls** resolve in priority order: local definitions, then imported or module-level names, then `self`/`cls` methods (walking resolved base classes), then attributes of imported modules or classes, and finally a conservative global-name fallback that links only when at most 3 definitions share the name. Builtins are never linked.

We evaluated the graph edge precision against the static analyzer Jedi. The `import`, `local`, and `self` edges are 100% accurate (220/220 agree) where Jedi resolves them, and the name fallback is 71% accurate (32/45), giving an overall precision of ~96%.

One Django snapshot yields 2,534 files, 8,358 classes, 23,347 functions and 30,495 call edges. It builds in 2.8 s cold (0.4 s with a warm cache) and serializes to 1.4 MB. 

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
- **Added lines that replace removed lines** inherit the removed lines' location.
- **Pure insertions** anchor to the preceding original line and are assigned to the deepest definition whose header is *less indented than the inserted code*.

On SWE-bench Lite, every reference patch edits exactly one file. 290 instances have at least one gold entity; the other 10 are module-level only and are scored at file level. 

## 4. Experiments

**Setup.** Queries are raw issue texts. Test files are excluded from the candidates. File-level BM25 indexes path tokens plus identifier-split contents. Entity-level BM25 ranks non-nested classes and functions within the top 10 files. **Acc@k** means *all* gold items are in the top k.

### 4.1 Lexical baseline

| Test set | File Acc@1 | File Acc@5 | Entity Acc@1 | Entity Acc@5 | Entity Acc@10 |
|---|---|---|---|---|---|
| Lite (n=290) | 43.3 | 70.0 | 19.7 | 36.9 | 47.6 |
| Verified (n=485) | 39.6 | 67.6 | 20.8 | 38.1 | 45.4 |

The massive gap between file-level and entity-level Acc@5 is the central observation. Lexical overlap usually finds the right *file*, but not the right *function* in it.

### 4.2 Graph-propagated retrieval

We run personalized PageRank from the top-k BM25 entities, weighted by score, over undirected `calls`, `inherits` and `contains` edges. Nested definitions are merged into their owners and tests are excluded. We select parameters on the **SWE-bench Lite dev split**. Its six repositories are disjoint from test. The selected setting is k=50, β=0.5, restart=0.5.

| Entity level | BM25 Acc@1 | PPR Acc@1 | BM25 Acc@5 | PPR Acc@5 | BM25 Acc@10 | PPR Acc@10 |
|---|---|---|---|---|---|---|
| Lite (n=290) | **19.7** | 17.6 | 36.9 | **39.0** | 47.6 | **48.6** |
| Verified (n=485) | **20.8** | 18.1 | **38.1** | 36.7 | 45.4 | **46.4** |

On Lite, propagation trades −2.1 at rank 1 for +2.1 at rank 5, but this is statistically insignificant (p=0.14). More importantly, the settings **fail to transfer to Verified**, dropping Acc@5 by −1.4. While passive propagation helps surface callers and callees at deeper cut-offs (Acc@10/R@10), it is not a reliable method overall. We hypothesize that graphs are more useful when agents *actively* follow their edges.

### 4.3 Auditing the competition's provided graphs and embeddings

The competition ships, for each of its 129 public dev tasks, a NetworkX code graph and 256-d node embeddings. These back the harness tools `get_code_neighbors`, `get_code_subgraph` and `search_similar_code`. We consume them locally and do not redistribute them.

**Protocol.** Gold entities come from our extractor. Provided node ids are mapped to ours by longest dotted-suffix match, and PPR reuses the Lite-dev configuration unchanged. We compare BM25; BM25+PPR over `codegraph`; the *same* BM25+PPR over the **provided** graph; and the provided embeddings used as the harness allows.

| Entity level, competition dev (n=121) | Acc@1 | Acc@5 | Acc@10 | Recall@10 |
|---|---|---|---|---|
| BM25 | 9.1 | 27.3 | 30.6 | 38.0 |
| BM25 + PPR over `codegraph` | 9.1 | 27.3 | **33.9** | **40.0** |
| BM25 + PPR over provided graph | 8.3 | 26.4 | 33.1 | 39.6 |
| Provided embeddings via issue symbols | 8.3 | 13.2 | 14.9 | 17.7 |

We also tested Gemma 4 E4B agents on these 129 dev tasks using different tools:

| Agent tools | Acc@1 | Acc@5 +fill | File Acc@1 | Unanswered |
|---|---|---|---|---|
| E4B, files | 24.0 | 33.9 | 44.2 | 26 |
| E4B, graph | **24.0** | **35.5** | **45.0** | **20** |
| E4B, provided (competition tools) | 19.0 | 31.4 | 41.1 | 36 |

**Findings.**
- **With the algorithm fixed, the provided graph is slightly weaker.** It has only call edges, lacks 8.9% of gold entities, and misses many library functions.
- **The embeddings are hard to use from issue text.** 46% of issues contain no resolvable symbol, making them nearly useless for zero-shot text search.
- **The organizers' tools are the weakest agent condition.** They significantly trail the standard graph tools (−5.0 Acc@1) and leave the most issues unanswered.

### 4.4 Graph tools for small Gemma 4 agents on a free T4

**Setup.** We serve Gemma 4 E4B and 12B (instruction-tuned, 4-bit QAT GGUF) with llama.cpp on Kaggle's free 2×T4: one server per GPU, 4 parallel slots, greedy decoding, thinking off. Both conditions share the same prompt and the same top-10 BM25 file hint:

- **(A) files:** `grep`, `list_dir`, `read_file`.
- **(B) graph:** (A) plus `search`, `outline`, `read_entity`, `callers`, `callees`.
- **(C) expanded:** Same as graph, but `search` and `read_entity` always list callers and callees inside their output.

We evaluate E4B on all 300 Lite and 500 Verified instances. 12B covers the full Lite 300 set. We report the agent's own list and a **+fill** variant that appends BM25's entity ranking after the agent's picks.

| Lite Test (entity n) | Acc@1 | Acc@5 +fill | File Acc@1 | Unanswered |
|---|---|---|---|---|
| BM25 (290) | 19.7 | 36.9 | 43.3 | – |
| E4B, files (290) | 29.0 | 49.0 | 63.0 | 39 |
| E4B, graph (290) | 32.8 | 50.7 | 67.0 | 32 |
| E4B, expanded (290) | 33.1 | 50.7 | 66.0 | 31 |
| 12B, files (282) | 39.4 | 53.5 | 73.9 | 35 |
| 12B, graph (282) | **45.7** | **58.2** | **78.0** | 19 |

| Verified Test (entity n) | Acc@1 | Acc@5 +fill | File Acc@1 | Unanswered |
|---|---|---|---|---|
| BM25 (484) | 20.9 | 38.2 | 39.5 | – |
| E4B, files (484) | 32.9 | 51.0 | 63.3 | 60 |
| E4B, graph (484) | **36.6** | **50.8** | **64.5** | **38** |

**Findings** (paired bootstrap, 95% CI):

1. **Agents beat passive retrieval.** E4B massive outperforms lexical BM25 (+13.1 Acc@1 on Lite, p=0.000). Model size is powerful: 12B adds another +12.9 over E4B.
2. **Graph tools help consistently.** Across models and datasets, active graph use provides a modest but consistent edge over file tools (+3.8 on Lite E4B, +6.3 on Lite 12B, +3.7 on Verified E4B). Pooled across both datasets (n=774), graph tools beat file tools by +4.5 Acc@1 (p=0.084).
3. **Graph tools fix hallucinations.** Failure taxonomy reveals 12B with file tools hallucinated nonexistent entity names on 10.3% of tasks; graph tools reduced this to 4.1%.
4. **Agents use the graph as a symbol index.** Agents frequently used `search` and `read_entity` to avoid grep-then-read loops, but they almost never navigated edges manually (e.g. `callers` was used 9 times, `callees` 0 times). Forcing the "expanded" condition showed no significant gain.
5. **The bottleneck is recognizing the target.** Hop analysis shows that in ~25% of failed E4B runs, the agent *was shown the gold function's code* but still didn't pick it (18–21% for 12B). Simply finding the file is not enough; the model must understand the defect.

## 5. Limitations

- **Static call resolution is approximate.** Dynamic dispatch and duck typing are partly missed; every edge is labelled with its resolution type so users can filter.
- **Python only for now.** Only the grammar and resolution rules are language-specific.
- **Graph Builder bug.** v1 graphs had a bug where builtins and `os.path.join` were incorrectly resolved. This was fixed in v2. All writeup numbers use v2.
- **Localization is necessary, not sufficient,** for repair. Its advantage is that it can be evaluated anywhere in seconds.

## 6. Reproducibility

- **Code:** https://github.com/Benadict-Git/codegraph-loc (Apache-2.0)
- **Dataset:** https://www.kaggle.com/datasets/benadictinfanta/codegraph-loc-swebench-lite
- **Reproduction notebook** (CPU; rebuilds all Lite tables and re-runs the pipeline live): https://www.kaggle.com/code/benadictinfanta/codegraph-loc-reproduction

```bash
pip install -e ".[dev,swebench]" && pytest   # 29 tests, < 1 s
python -m cgloc.eval.run_bm25                # §4.1
python -m cgloc.data.build_dataset           # graphs + gold, ~10 min CPU
python -m cgloc.eval.run_graph_rank          # §4.2, dev selection -> test
python -m cgloc.eval.run_provided            # §4.3, needs competition data
python kaggle/push_agent.py <name> '<config>' # §4.4 on a Kaggle T4
python -m cgloc.agent.rescore <runs.jsonl>   # §4.4 scoring from saved raw answers
python -m cgloc.eval.agent_report a=... b=... # §4.4 tables + bootstrap CIs
```

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
