"""Write the public reproduction notebook (codegraph-loc-reproduction.ipynb) next to this file."""
import json
from pathlib import Path

CELLS = [
    ("md", """# CodeGraph-Loc: reproduction notebook

This notebook reproduces the SWE-bench Lite results of the Gemma 4 Developer Agent Paper Track writeup
*CodeGraph-Loc: Laptop-Scale Code Graphs for Small-Agent Bug Localization*, using only the public
[CodeGraph-Loc dataset](https://www.kaggle.com/datasets/benadictinfanta/codegraph-loc-swebench-lite)
and the [code on GitHub](https://github.com/Benadict-Git/codegraph-loc). It runs on CPU.

1. Dataset scale (§3.1)
2. BM25 baseline (§4.1) and graph-propagated retrieval (§4.2)
3. Gemma 4 agents with file vs graph tools, with paired bootstrap CIs (§4.4)
4. The `codegraph` query API on one instance
5. A live end-to-end re-run (clone, extract gold, build graph, BM25) on the six `psf/requests` instances,
   checked against the saved results

§4.3 (the audit of the competition's provided graphs) needs the competition data, which may not be
redistributed; it is reproducible with `python -m cgloc.eval.run_provided` after downloading that data."""),
    ("code", "!pip install -q git+https://github.com/Benadict-Git/codegraph-loc"),
    ("code", """import glob, json, os, statistics as st
from collections import defaultdict
DATA = os.path.dirname(glob.glob('/kaggle/input/**/instances.jsonl', recursive=True)[0])
print(DATA, sorted(os.listdir(DATA)))
read = lambda p: [json.loads(l) for l in open(os.path.join(DATA, p)) if l.strip()]"""),
    ("md", "## 1. Dataset scale (§3.1)"),
    ("code", """stats = read('stats.jsonl')
tot = lambda k: sum(s.get(k, 0) for s in stats)
med = lambda k: st.median(s.get(k, 0) for s in stats)
print(f"instances: {len(stats)}")
for k in ['files', 'nodes', 'edges_contains', 'edges_calls', 'edges_imports', 'edges_inherits', 'secs']:
    print(f"{k:16} total {tot(k):>12,.0f}   median {med(k):>9,.1f}")
inst = read('instances.jsonl')
print('instances with >=1 gold entity:', sum(bool(i['gold']['entities']) for i in inst),
      '| single-file patches:', sum(len(i['gold']['files']) == 1 for i in inst))"""),
    ("md", "## 2. BM25 (§4.1) and BM25 + personalized PageRank (§4.2)"),
    ("code", """from cgloc.eval.metrics import aggregate
bm = read('results/bm25_lite.jsonl')
pct = lambda d: {k: round(v * 100, 1) for k, v in d.items() if k.startswith('acc@')}
print('file  ', pct(aggregate(r['file'] for r in bm)))
print('entity', pct(aggregate(r['entity'] for r in bm)))
by = defaultdict(list)
for r in bm: by[r['repo']].append(r)
print('\\nAcc@5 by repository (file / entity):')
for repo, rs in sorted(by.items(), key=lambda x: -len(x[1])):
    e = [r['entity']['acc@5'] for r in rs if r['entity']]
    print(f"  {repo:28} n={len(rs):3}  {100*sum(r['file']['acc@5'] for r in rs)/len(rs):5.1f}  {100*sum(e)/max(len(e),1):5.1f}")
gr = json.load(open(os.path.join(DATA, 'results/graph_rank_lite.json')))
print('\\nselected on dev:', gr['selected_on_dev'])
print('test BM25      ', pct(gr['test_bm25']))
print('test BM25 + PPR', pct(gr['test_selected']))
cfgs = [k for k in gr['test'] if k != 'bm25']
print(f"grid settings beating BM25 on test Acc@5: {sum(gr['test'][k]['acc@5'] > gr['test_bm25']['acc@5'] for k in cfgs)}/{len(cfgs)}")"""),
    ("md", "## 3. Gemma 4 agents on a free T4 (§4.4)\nE4B covers all 300 instances; 12B covers a 144-instance subset (see the writeup)."),
    ("code", """from cgloc.eval.agent_report import load, report
A = os.path.join(DATA, 'results/agents')
bm25 = load(os.path.join(DATA, 'results/bm25_lite.jsonl'))
print(report(bm25, {'e4b_files': load(f'{A}/lite_e4b_files.jsonl'), 'e4b_graph': load(f'{A}/lite_e4b_graph.jsonl')}))
print()
print(report(bm25, {'12b_files': load(f'{A}/lite_12b_files.jsonl'), '12b_graph': load(f'{A}/lite_12b_graph.jsonl')}))"""),
    ("code", """# One full trajectory (graph condition)
r = next(iter(load(f'{A}/lite_e4b_graph.jsonl').values()))
print(r['instance_id'], '| gold:', r['gold']['entities'], '| predicted:', r['pred_entities'])
for s in r['steps']:
    print(' ', s.get('tool', 'FINAL'), json.dumps(s.get('args', s.get('final')))[:120])"""),
    ("md", "## 4. The `codegraph` API"),
    ("code", """from codegraph import CodeGraph, load_graph
cg = CodeGraph(load_graph(os.path.join(DATA, 'graphs/django__django-11099.json')))
print(len(cg.nodes), 'nodes')
for h in cg.search('UsernameValidator', limit=5): print(' ', h['id'], h['kind'])
target = cg.search('ASCIIUsernameValidator', kind='class')[0]['id']
print('bases of', target, '->', cg.bases(target))
print('outline:'); [print('  ' * lvl, n['id'].split('::')[1]) for lvl, n in cg.outline('django/contrib/auth/validators.py')]"""),
    ("md", "## 5. Live re-run on `psf/requests` (clone, gold, graph, BM25)"),
    ("code", """from pathlib import Path
from cgloc.data.swebench import load_instances
from cgloc.eval.run_bm25 import run_instance
saved = {r['instance_id']: r for r in bm}
for inst in [i for i in load_instances('lite') if i['repo'] == 'psf/requests']:
    row = run_instance(inst, Path('/tmp/repos'))
    old = saved[inst['instance_id']]
    same = row['gold'] == old['gold'] and row['pred_entities'][:10] == old['pred_entities'][:10]
    print(f"{inst['instance_id']:22} gold={row['gold']['entities']}  matches saved: {same}")"""),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).with_name("codegraph-loc-reproduction.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
