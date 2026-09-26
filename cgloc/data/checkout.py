from __future__ import annotations

import subprocess
from pathlib import Path

_FILE_MODES = {"100644", "100755"}


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def ensure_clone(repo: str, cache_dir: str | Path) -> Path:
    """Bare clone of github.com/<repo> cached under cache_dir."""
    dest = Path(cache_dir) / repo.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--bare", "--quiet", f"https://github.com/{repo}.git", str(dest))
    return dest


def ensure_commit(clone: Path, commit: str) -> None:
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}", cwd=clone)
    except subprocess.CalledProcessError:
        _git("fetch", "--quiet", "origin", commit, cwd=clone)


def ls_tree(clone: Path, commit: str, suffix: str = ".py") -> dict[str, str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--full-tree", "-z", commit], cwd=clone,
                         check=True, capture_output=True).stdout
    files = {}
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, sha = meta.decode().split()
        p = path.decode("utf-8", "replace")
        if kind == "blob" and mode in _FILE_MODES and p.endswith(suffix):
            files[p] = sha
    return files


class BlobReader:
    """Streams blobs through a single `git cat-file --batch` process."""

    def __init__(self, clone: Path):
        self.proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=clone,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, sha: str) -> bytes | None:
        self.proc.stdin.write(f"{sha}\n".encode())
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            return None
        data = self.proc.stdout.read(int(header[2]))
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> BlobReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_files(clone: Path, commit: str, suffix: str = ".py") -> dict[str, bytes]:
    tree = ls_tree(clone, commit, suffix)
    with BlobReader(clone) as reader:
        return {p: data for p, sha in sorted(tree.items()) if (data := reader.read(sha)) is not None}
