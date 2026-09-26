MODELS = '''\
import os
from pkg import utils
from .utils import slugify as make_slug


class Base:
    def save(self):
        return self.validate()

    def validate(self):
        return True


class Model(Base):
    name = "x"

    @property
    def slug(self):
        return make_slug(self.name)

    def save(self):
        self.validate()
        utils.log("saved")
        return os.path.join("a", "b")

    def outer(self):
        def inner():
            return helper()
        return inner()


def helper():
    return Model()
'''

SOURCES = {
    "pkg/__init__.py": "from .models import Model\n",
    "pkg/models.py": MODELS,
    "pkg/utils.py": "def slugify(value):\n    return value.lower()\n\n\ndef log(msg):\n    print(msg)\n",
    "src/other/__init__.py": "",
    "src/other/api.py": "from other.core import run\n\n\ndef main():\n    run()\n",
    "src/other/core.py": "def run():\n    pass\n",
    "tests/test_models.py": (
        "from pkg import Model as Reexported\n"
        "from pkg.models import Model\n\n\n"
        "def test_save():\n    Model().save()\n\n\n"
        "def test_reexport():\n    return Reexported()\n"
    ),
}


def sources_bytes() -> dict[str, bytes]:
    return {k: v.encode() for k, v in SOURCES.items()}
