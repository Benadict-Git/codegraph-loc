from codegraph import CodeGraph, build_graph, load_graph, save_graph
from codegraph.build import module_name
from fixture_repo import MODELS, SOURCES, sources_bytes

M = "pkg/models.py"


def _graph():
    return build_graph(sources_bytes())


def _edges(graph, kind):
    return {(s, d) for s, d, k, _ in graph["edges"] if k == kind}


def test_nodes_and_spans():
    nodes = {n["id"]: n for n in _graph()["nodes"]}
    lines = MODELS.splitlines()
    assert nodes[f"{M}::Model.slug"]["start"] == lines.index("    @property") + 1
    assert nodes[f"{M}::Model"]["kind"] == "class"
    assert nodes[f"{M}::Model.outer.inner"]["nested"] is True
    assert nodes[f"{M}::Model.save"]["nested"] is False
    assert nodes["tests/test_models.py"]["test"] is True
    assert nodes[M]["test"] is False
    assert nodes[M]["end"] == len(lines)


def test_module_names_handle_src_layout():
    pkg_dirs = {"pkg", "src/other"}
    assert module_name("pkg/__init__.py", pkg_dirs) == "pkg"
    assert module_name("src/other/api.py", pkg_dirs) == "other.api"
    assert module_name("tests/test_models.py", pkg_dirs) == "test_models"


def test_imports():
    imports = _edges(_graph(), "imports")
    assert (M, "pkg/utils.py") in imports
    assert ("pkg/__init__.py", M) in imports
    assert ("src/other/api.py", "src/other/core.py") in imports
    assert ("tests/test_models.py", M) in imports


def test_inheritance_and_contains():
    g = _graph()
    assert (f"{M}::Model", f"{M}::Base") in _edges(g, "inherits")
    contains = _edges(g, "contains")
    assert (M, f"{M}::Model") in contains
    assert (f"{M}::Model", f"{M}::Model.save") in contains
    assert (f"{M}::Model.outer", f"{M}::Model.outer.inner") in contains


def test_call_resolution():
    calls = _edges(_graph(), "calls")
    expected = {
        (f"{M}::Base.save", f"{M}::Base.validate"),
        (f"{M}::Model.slug", "pkg/utils.py::slugify"),
        (f"{M}::Model.save", f"{M}::Base.validate"),
        (f"{M}::Model.save", "pkg/utils.py::log"),
        (f"{M}::Model.outer", f"{M}::Model.outer.inner"),
        (f"{M}::Model.outer.inner", f"{M}::helper"),
        (f"{M}::helper", f"{M}::Model"),
        ("src/other/api.py::main", "src/other/core.py::run"),
        ("tests/test_models.py::test_reexport", f"{M}::Model"),
        ("tests/test_models.py::test_save", f"{M}::Model"),
    }
    assert expected <= calls
    assert not any(d.endswith("::join") or d.endswith("::print") for _, d in calls)


def test_parse_cache_reused():
    cache = {}
    build_graph(sources_bytes(), cache=cache)
    n = len(cache)
    build_graph(sources_bytes(), cache=cache)
    assert len(cache) == n == len(set(SOURCES.values()))


def test_roundtrip_and_query(tmp_path):
    path = tmp_path / "g.json.gz"
    save_graph(_graph(), path)
    cg = CodeGraph(load_graph(path), SOURCES.get)
    assert cg.search("slugify")[0]["id"] == "pkg/utils.py::slugify"
    assert cg.search("Model.save")[0]["id"] == f"{M}::Model.save"
    assert all(not n["test"] for n in cg.search("test"))
    assert f"{M}::Model.save" in cg.callers(f"{M}::Base.validate")
    assert cg.bases(f"{M}::Model") == [f"{M}::Base"]
    assert cg.subclasses(f"{M}::Base") == [f"{M}::Model"]
    assert cg.locate(M, MODELS.splitlines().index('        utils.log("saved")') + 1) == f"{M}::Model.save"
    assert cg.locate(M, 1) == M
    assert "def slugify(value):" in cg.read("pkg/utils.py::slugify")
    outline = [n["id"] for _, n in cg.outline(M)]
    assert outline[:3] == [f"{M}::Base", f"{M}::Base.save", f"{M}::Base.validate"]
    assert f"{M}::Model.outer.inner" not in outline
    hop1 = dict(cg.neighbors(f"{M}::Model.save"))
    assert hop1[f"{M}::Model"] == 1 and hop1["pkg/utils.py::log"] == 1
