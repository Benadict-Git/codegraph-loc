import json

from codegraph import CodeGraph, build_graph
from cgloc.agent.llm import Reply
from cgloc.agent.loop import extract_json, run_episode
from cgloc.agent.run import normalize
from cgloc.agent.tools import Workspace, call_tool, file_tools, graph_tools
from fixture_repo import MODELS, sources_bytes

MODELS_LINES = MODELS.splitlines()

M = "pkg/models.py"


def _env():
    ws = Workspace(sources_bytes())
    cg = CodeGraph(build_graph(ws.sources), ws.text)
    return ws, cg, {t.name: t for t in file_tools(ws) + graph_tools(cg)}


def test_extract_json():
    assert extract_json('noise ```json\n{"tool": "grep", "args": {"pattern": "a{2}\\\\\\""}}\n``` end') == \
        {"tool": "grep", "args": {"pattern": 'a{2}\\"'}}
    assert extract_json('{bad} then {"final": ["x"]}') == {"final": ["x"]}
    assert extract_json("no json") is None


def test_tools():
    _, _, tools = _env()
    assert "pkg/utils.py:1: def slugify(value):" in call_tool(tools, "grep", {"pattern": "def slugify"})
    assert "tests/" not in call_tool(tools, "grep", {"pattern": "Model"})
    assert call_tool(tools, "list_dir", {"path": "pkg"}).splitlines() == ["__init__.py", "models.py", "utils.py"]
    assert "    2     return value.lower()" in call_tool(tools, "read_file", {"path": "pkg/utils.py"})
    assert f"{M}::Model.save" in call_tool(tools, "callers", {"node": "Base.validate"})
    assert "pkg/utils.py::log" in call_tool(tools, "callees", {"node": f"{M}::Model.save"})
    assert "  Model.save" in call_tool(tools, "outline", {"path": M})
    assert "def slugify(value):" in call_tool(tools, "read_entity", {"node": "slugify"})
    assert call_tool(tools, "nope", {}).startswith("unknown tool")
    assert call_tool(tools, "grep", {"wrong": 1}).startswith("bad arguments")


def test_episode_with_scripted_llm():
    ws, cg, _ = _env()
    script = iter([
        '{"thought": "look", "tool": "search", "args": {"query": "slugify"}}',
        "I am not sure",
        '{"thought": "done", "final": ["pkg/utils.py::slugify", "Model.slug"]}',
    ])
    seen = []

    def llm(messages):
        seen.append(messages[-1]["content"])
        return Reply(next(script), 10, 5, 0.01)

    ep = run_episode("slug is wrong", "demo/repo", file_tools(ws) + graph_tools(cg), llm, budget=3)
    assert ep["final"] == ["pkg/utils.py::slugify", "Model.slug"]
    assert ep["tool_calls"] == 1 and ep["invalid"] == 1
    assert "pkg/utils.py::slugify" in seen[1] and "[2 tool calls left]" in seen[1]
    assert ep["prompt_tokens"] == 30
    json.dumps(ep)


def test_budget_forces_final_answer():
    ws, cg, _ = _env()
    replies = iter(['{"tool": "grep", "args": {"pattern": "x"}}'] * 2 + ['{"final": ["pkg/utils.py::log"]}'])
    ep = run_episode("x", "r", file_tools(ws), lambda m: Reply(next(replies), 1, 1, 0.0), budget=1)
    assert ep["final"] == ["pkg/utils.py::log"] and ep["steps"][-1].get("forced")


def test_normalize_answers():
    _, cg, _ = _env()
    ents, files = normalize(["`pkg/models.py::Model.save`", "Model.outer.inner", "slugify()", "pkg/utils.py",
                             "pkg/models.py::Model.missing", "nonsense_zzz"], cg)
    assert ents == [f"{M}::Model.save", f"{M}::Model.outer", "pkg/utils.py::slugify"]
    assert files[:2] == [M, "pkg/utils.py"]


def test_parse_final_variants():
    from cgloc.agent.loop import parse_final
    assert parse_final({"final": ["a", "b"]}) == ["a", "b"]
    assert parse_final({"tool": "final", "args": {"locations": ["a"]}}) == ["a"]
    assert parse_final({"tool": "Final", "args": {"answer": "a"}}) == ["a"]
    assert parse_final({"tool": "submit", "args": ["a"]}) == ["a"]
    assert parse_final({"tool": "final", "args": {}}) == []
    assert parse_final({"tool": "grep", "args": {"pattern": "x"}}) is None
    assert parse_final(None) is None


def test_final_submitted_as_tool_ends_episode():
    ws, cg, _ = _env()
    replies = iter(['{"tool": "final", "args": {"locations": ["pkg/utils.py::log"]}}'])
    ep = run_episode("x", "r", file_tools(ws), lambda m: Reply(next(replies), 1, 1, 0.0), budget=3)
    assert ep["final"] == ["pkg/utils.py::log"] and ep["tool_calls"] == 0


def test_normalize_line_refs_and_wrong_prefix():
    _, cg, _ = _env()
    line = MODELS_LINES.index('        utils.log("saved")') + 1
    ents, _ = normalize([f"{M}::{line}", f"{M}:L{line}", "pkg/models.py::Wrong.validate", "pkg/models.py::slug"], cg)
    assert ents == [f"{M}::Model.save", f"{M}::Base.validate", f"{M}::Model.slug"]
