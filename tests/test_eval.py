import subprocess

import pytest

from cgloc.data.checkout import BlobReader, ls_tree, read_files
from cgloc.eval.bm25 import rank_entities, rank_files, tokenize
from cgloc.eval.metrics import aggregate, rank_metrics
from fixture_repo import sources_bytes


def test_rank_metrics():
    m = rank_metrics(["a", "b", "c"], {"b", "c"}, ks=(1, 3))
    assert m == {"acc@1": 0.0, "recall@1": 0.0, "acc@3": 1.0, "recall@3": 1.0, "mrr": 0.5}
    assert rank_metrics(["a"], set()) is None
    agg = aggregate([m, None, rank_metrics(["b"], {"b"}, ks=(1, 3))])
    assert agg["n"] == 2 and agg["acc@1"] == 0.5


def test_tokenize_splits_identifiers():
    toks = tokenize("QuerySet.get_or_create HTTPServer")
    assert {"query", "set", "queryset", "get", "or", "create", "get_or_create", "http", "server"} <= set(toks)


def test_bm25_ranks_relevant_file_and_entity():
    src = sources_bytes()
    query = "slugify should not lower-case the value"
    assert rank_files(query, src)[0] == "pkg/utils.py"
    assert "tests/test_models.py" not in rank_files(query, src)
    assert rank_entities(query, src)[0] == "pkg/utils.py::slugify"


def test_graph_rerank_pulls_in_callee_outside_lexical_hits():
    from codegraph import build_graph
    from cgloc.eval.graph_rank import entity_adjacency, graph_rerank, personalized_pagerank

    adj, entities = entity_adjacency(build_graph(sources_bytes()))
    M = "pkg/models.py"
    assert f"{M}::Model.outer" in adj[f"{M}::helper"]
    assert not any("tests/" in e for e in entities) and f"{M}::Model.outer.inner" not in entities
    p = personalized_pagerank(adj, {f"{M}::Model.slug": 1.0})
    assert abs(sum(p.values()) - 1) < 1e-9
    ranked = graph_rerank({f"{M}::Model.slug": 3.0}, adj, entities, top_k=1, beta=0.5)
    assert ranked[0] == f"{M}::Model.slug" and "pkg/utils.py::slugify" in ranked[:3]


@pytest.fixture
def git_repo(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path,
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/a.py").write_text("def f():\n    pass\n")
    (tmp_path / "README.md").write_text("hi\n")
    git("add", ".")
    git("commit", "-qm", "init")
    return tmp_path, git("rev-parse", "HEAD")


def test_blob_access(git_repo):
    repo, commit = git_repo
    tree = ls_tree(repo, commit)
    assert list(tree) == ["pkg/a.py"]
    with BlobReader(repo) as r:
        assert r.read(tree["pkg/a.py"]) == b"def f():\n    pass\n"
        assert r.read("0" * 40) is None
    assert read_files(repo, commit) == {"pkg/a.py": b"def f():\n    pass\n"}
