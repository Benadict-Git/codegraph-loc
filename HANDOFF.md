# Handoff: CodeGraph-Loc (Gemma 4 Paper Track), continue from here

> For the agent continuing this work (e.g. Antigravity). Read this whole file first. It replaces the older `summary.md` (kept only for history).
> State as of **2026-10-10**. Owner: Benadict (Kaggle `benadictinfanta`, GitHub `Benadict-Git`), who is preparing for the GATE exam and wants the agent to do the work.

## 1. Goal and deadlines
- **Competition:** Kaggle **"Google - The Gemma 4 Developer Agent Paper Track"** (`gemma-4-developer-agent-paper`). Prizes: Best Paper $15k, Best New Resource $10k, Best New Application $10k.
- **Writeup v1 is already SUBMITTED** and editable until **2026-11-12 23:59 UTC**. The limit is **3,000 words**.
- **Rubric:** novelty, quality, relevance, verifiability, clarity, each scored 0–5 and averaged.
- **Main leaderboard competition** (`gemma-4-developer-agent`): handled separately by another agent in `~/Projects/gemma-agent-main/` (brief `AGY_PROMPT.md`). **Do not touch that workspace.**

## 2. Safety and rules (must follow)
- **Work only inside** `/home/benny/Projects/google-gemma/`.
  - Never modify other folders in `~/Projects/` or dotfiles.
  - Never print or copy credentials (`KAGGLE_API_TOKEN`, `~/.kaggle/`, gh tokens).
  - No `sudo`, no global installs. Use `.venv/`.
- **Never redistribute competition data** (`cache/kaggle/main/`: graphs, embeddings, tasks). Use it only locally or in **private** Kaggle kernels.
  - The public dataset must contain only SWE-bench-derived data.
- **Before publishing anything new, ask the user.** Updating the existing public dataset and notebook with new SWE-bench results was approved.
- **Never delete or modify the user's other Kaggle notebooks** (the `g4agent-*`, `umud-*` and `rsna-*` ones). Only manage items whose slug starts with `cgloc-`, plus `codegraph-loc-reproduction` and `codegraph-loc-swebench-lite`.
- **Report results honestly,** including negative ones. Never state a number you didn't measure.

## 3. Published assets
| Asset | Location |
|---|---|
| Code (Apache-2.0) | https://github.com/Benadict-Git/codegraph-loc. Local branch `main` is **2 commits ahead** of origin (unpushed: `cb3beef`, `1363f0c`) |
| Public dataset (v1 graphs) | https://www.kaggle.com/datasets/benadictinfanta/codegraph-loc-swebench-lite |
| Public reproduction notebook | https://www.kaggle.com/code/benadictinfanta/codegraph-loc-reproduction (source: `kaggle/repro_notebook/build_notebook.py`) |
| Private code dataset for GPU jobs | `benadictinfanta/cgloc-code` (sync with `bash kaggle/sync_code.sh "msg"`) |
| Writeup text pasted on Kaggle | `paper/kaggle_writeup.md` (2,993 words); long copy `paper/writeup.md`; card `paper/card/card_1120x560.png` |

## 4. Repo map
- `codegraph/`: tree-sitter parser (`parse.py`), graph builder with call resolution (`build.py`), query API (`query.py`).
- `cgloc/data/`: SWE-bench loading, git-blob access, gold-location extraction (`gold.py`), dataset builder.
- `cgloc/eval/`: the analysis scripts.
  - Retrieval: `bm25.py`, `run_bm25.py`, `graph_rank.py`, `run_graph_rank.py` (`--config` = fixed setting with per-instance output and bootstrap).
  - Competition audit: `run_provided.py` (§4.3).
  - Agent analysis: `agent_report.py` (tables + paired bootstrap), `hop_analysis.py`, `edge_audit.py` (vs Jedi), `failure_taxonomy.py`.
- `cgloc/agent/`: localization agent.
  - `tools.py`: file / graph / **expanded** / **provided** tool sets.
  - `run.py`: conditions `files|graph|expanded|provided`.
  - `rescore.py`: re-scores saved raw answers.
- `kaggle/`: GPU jobs.
  - `push_agent.py <name> '<json cfg>'` generates and pushes a private T4×2 kernel from `agent_run/run_agent.py`, which runs llama.cpp, one server per GPU, with a watchdog.
  - `smoke_llamacpp/` holds the llama.cpp CUDA build (its output provides the binaries).
- `tests/`: **34 tests**; run `.venv/bin/python -m pytest -q`.

## 5. What changed in Phase 2 (2026-10-05)
- **Graph-builder bug fixed (v2 graphs).**
  - `os.path.join()` was being treated as a plain `join()` call.
  - Builtin methods (`list.extend` and similar) were linked to same-named repo methods.
  - Fixed in commit `1363f0c`. **All writeup numbers that depend on call edges must use v2 results.**
  - Agent runs with plain graph tools are unaffected: they use `search`/`read_entity`/`outline`, and `callers` was called only 9 times.
- **New conditions:**
  - **expanded**: `read_entity`/`search` always list callers and callees.
  - **provided**: emulation of the competition's `get_code_neighbors`, `search_similar_code` and `get_code_subgraph`.

## 6. Results so far (all measured; details in `outputs/v2/RESULTS.md`)
- **§4.2 passive graph re-ranking (v2):**
  - **SWE-bench Lite (n=290):** BM25 Acc@1/5/10 = 19.7 / 36.9 / 47.6; PPR = 17.6 / 39.0 / 48.6. Acc@5 +2.1 [−0.3, +4.5], p=0.14, not significant.
  - **SWE-bench Verified (n=485, no retuning):** Acc@5 −1.4, Acc@1 −2.7 [−5.2, −0.2].
  - So passive propagation is not a reliable win. **The current writeup overclaims this ("tuning-robust gain"), and §4.2 must be rewritten.**
- **C, edge audit vs Jedi:** `import`/`local`/`self` edges agree **220/220** where Jedi resolves; name fallback **71%** (32/45), which is 15% of call edges. ≈ **96%** precision overall.
- **D, Verified BM25:** file Acc@1/5/10 = 39.6 / 67.6 / 76.6; entity 20.8 / 38.1 / 45.4. Graphs and gold are built in `outputs/dataset/verified/`.
- **§4.3 provided audit:** same numbers on v2.
- **F, agents on the 129 competition dev tasks (E4B; entity n=121):**

  | Run | Acc@1 | Acc@5 +fill | File Acc@1 | Unanswered |
  |---|---|---|---|---|
  | BM25 | 9.1 | 27.3 | 16.3 | – |
  | files | 24.0 | 33.9 | 44.2 | 26 |
  | graph | 24.0 | 35.5 | 45.0 | 20 |
  | **provided** (organizers' tools) | **19.0** | 31.4 | 41.1 | 36 |

  All agents beat BM25 (+14.9 Acc@1). **The organizers' tools are the weakest agent condition** (−5.0 vs graph, p=0.13) and leave the most issues unanswered.
- **A, hop analysis (Lite):** among E4B misses, in **~25%** of issues the agent *was shown the gold function's code* and still didn't answer it; ~12% were 1 hop away and ~10% 2 hops away. For 12B: 18–21% shown, 8–9% 1 hop.
- **I, failure taxonomy:** about 21% of issues end at the right file but the wrong function, for every model and tool set. 12B mainly cuts wrong-file errors. Graph tools cut invented names for 12B (7.9% → 0.7%). See `outputs/v2/failure_taxonomy.json`.
- **E, expanded (v1 graphs):** Acc@1 35.2 vs graph 32.8 vs files 29.0; file Acc@1 71.0. **The v2 rerun has finished on Kaggle but is not downloaded or scored yet.**

## 7. Remaining work (in order)
1. **Download the finished Kaggle jobs:**
   ```bash
   for k in agent-e4b-expanded agent-12b-rest agent-e4b-verified; do mkdir -p outputs/kaggle/$k; kaggle kernels output benadictinfanta/cgloc-$k -p outputs/kaggle/$k; done
   ```
   Check each log for `FAILED` lines; expect 0. 12B may show watchdog "restarting" lines, which are fine.
2. **Rescore** (always score from saved raw answers):
   ```bash
   .venv/bin/python -m cgloc.agent.rescore outputs/kaggle/agent-e4b-expanded/lite_e4b_expanded.jsonl \
       outputs/kaggle/agent-12b-rest/lite_12b_files_rest.jsonl outputs/kaggle/agent-12b-rest/lite_12b_graph_rest.jsonl
   .venv/bin/python -m cgloc.agent.rescore outputs/kaggle/agent-e4b-verified/verified_e4b_files.jsonl \
       outputs/kaggle/agent-e4b-verified/verified_e4b_graph.jsonl --dataset verified
   ```
3. **Reports** (`agent_report.py` takes `name=path` pairs, compared in order, with a paired bootstrap):
   - **Lite E4B:** files vs graph vs **expanded** (v2).
   - **12B full 300:** concatenate the 144-subset runs (`outputs/kaggle/agent-12b/lite_12b_files.rescored.jsonl`, `outputs/kaggle/agent-12b-graph/lite_12b_graph.rescored.jsonl`) with the `*_rest` runs into one file per condition, then report vs BM25 (`outputs/bm25_lite.jsonl`).
   - **Verified:** `--bm25 outputs/v2/bm25_verified.jsonl`, files vs graph.
   - **Pooled graph-vs-files** on Lite + Verified (larger n) for statistical power.
   - Re-run `failure_taxonomy` and `hop_analysis` including the new runs.
4. **Rewrite `paper/kaggle_writeup.md` (≤ 3,000 words; check with `wc -w`)** and copy it to `paper/writeup.md` with the title and subtitle header.
   - **§4.2:** honest v2 numbers plus CIs and the Verified non-transfer (passive graph use doesn't help reliably).
   - **§3:** add the edge-precision audit, and the Verified extension to the resource (785 instances total).
   - **§4.3:** add agent-level F results; the organizers' tools are the weakest condition.
   - **§4.4:** add expanded (v2), 12B on the full 300, and Verified; add the hop analysis and failure taxonomy as short findings.
   - Main story: **graphs help when agents use them actively (and especially when edges are shown automatically), not as a passive re-ranker.**
   - Make room by cutting: per-repo BM25 table, related-work wording, the oracle row.
   - Update the abstract, the contributions and the limitations. Mention v2 graphs and the bug fix briefly in limitations or reproducibility.
5. **Dataset v2** (same public dataset):
   - Stage `kaggle/.public_dataset/data/`: Lite v2 graphs (`outputs/dataset/lite/`), plus Verified graphs and gold (`outputs/dataset/verified/`, under a `verified/` subfolder), plus new results (v2 PPR, Verified BM25/PPR, all rescored agent runs, edge audit, hop analysis, failure taxonomy).
   - **Do not include `outputs/kaggle/agent-e4b-comp/*` or anything from `cache/kaggle/main/`**: those are competition data or derived per-task outputs.
   - Then run `kaggle datasets version -p kaggle/.public_dataset -r zip -m "v2: fixed graphs, Verified, new runs"` and update the description in `dataset-metadata.json`.
   - Note: the Kaggle zip upload strips a single top-level folder and decompresses `.gz` files; `load_graph` handles both.
6. **Notebook:** extend `kaggle/repro_notebook/build_notebook.py` with the new tables, then run `.venv/bin/python kaggle/repro_notebook/build_notebook.py && kaggle kernels push -p kaggle/repro_notebook`. It must finish with 0 errors; verify with `kaggle kernels output`.
7. **Push to GitHub:** run `git -c credential.helper= -c credential.helper='!gh auth git-credential' push` (plain `git push` has no credentials). Update the README test count and links.
8. **Tell the user** to re-paste `paper/kaggle_writeup.md` into the Kaggle writeup (Edit → Project Description → paste → Save) and to add any new links. That's about 10 minutes of their time.

## 8. Kaggle and compute notes
- **GPU quota:** 30 h/week (2.3 h used this week; refresh 2026-10-17). `kaggle quota` shows usage.
- **The account allows only 2 concurrent batch GPU sessions.** If a push fails with "Maximum batch GPU session count of 2 reached", retry later.
  - The other agent's `g4agent-*` jobs share this limit. Don't touch them; ask the user.
  - A stuck queued run can only be cancelled in the Kaggle UI. The CLI can't cancel, and deleting a notebook is destructive.
- **Background processes:** long background shell tasks get killed. Launch long CPU jobs as `setsid nohup <cmd> > log 2>&1 < /dev/null &`, and note that the real PID is `$!`+1 because `setsid` forks.
- **CLI quirks:** `kaggle competitions pages <slug> list <slug> --content` needs the slug twice; `kaggle kernels push --accelerator NvidiaTeslaT4`.
- **12B on 2×T4:** two llama.cpp servers get host-RAM-OOM-killed. The watchdog restarts them and the client retries 7 times, so results are valid but 12B timings are not comparable.
- **Measured throughput (E4B, 2×T4):** ~55–65 s per issue with 8 parallel slots. 12B: ~220–400 s per issue.
