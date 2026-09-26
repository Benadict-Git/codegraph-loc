import difflib

from cgloc.data.gold import gold_locations
from fixture_repo import MODELS

M = "pkg/models.py"


def _patch(old: str, new: str, path: str = M) -> str:
    a = f"a/{path}" if old else "/dev/null"
    diff = difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=a, tofile=f"b/{path}")
    return f"diff --git a/{path} b/{path}\n" + "".join(diff)


def _gold(new: str):
    return gold_locations(_patch(MODELS, new), {M: MODELS.encode()}.get)


def test_modified_line_in_method():
    g = _gold(MODELS.replace('utils.log("saved")', 'utils.log("done")'))
    assert g.files == [M]
    assert g.entities == [f"{M}::Model.save"]
    assert g.module_level == []


def test_change_in_nested_function_maps_to_owner():
    g = _gold(MODELS.replace("return helper()", "return helper() or None"))
    assert g.entities == [f"{M}::Model.outer"]


def test_inserted_method_maps_to_class():
    new = MODELS.replace('        return os.path.join("a", "b")\n',
                         '        return os.path.join("a", "b")\n\n    def delete(self):\n        pass\n')
    assert _gold(new).entities == [f"{M}::Model"]


def test_inserted_line_at_end_of_function_body():
    new = MODELS.replace("        return self.validate()\n", "        return self.validate()\n        x = 1\n", 1)
    assert _gold(new).entities == [f"{M}::Base.save"]


def test_new_top_level_function_is_module_level():
    g = _gold(MODELS + "\n\ndef extra():\n    return 1\n")
    assert g.entities == [] and g.module_level == [M]


def test_decorator_change_maps_to_decorated_method():
    g = _gold(MODELS.replace("    @property\n", "    @cached_property\n"))
    assert g.entities == [f"{M}::Model.slug"]


def test_import_change_is_module_level():
    g = _gold(MODELS.replace("import os\n", "import os\nimport sys\n"))
    assert g.entities == [] and g.module_level == [M]


def test_added_file_and_multi_file_patch():
    patch = _patch(MODELS, MODELS.replace("return True", "return False")) + _patch("", "X = 1\n", "pkg/new.py")
    g = gold_locations(patch, {M: MODELS.encode()}.get)
    assert g.files == [M, "pkg/new.py"]
    assert g.added_files == ["pkg/new.py"]
    assert g.entities == [f"{M}::Base.validate"]
