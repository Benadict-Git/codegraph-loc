"""Generate and push a Kaggle agent job from kaggle/agent_run/ with a given run config.

usage: python kaggle/push_agent.py <job-name> '<json config>'
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER = "benadictinfanta"


def main() -> None:
    name, cfg = sys.argv[1], json.loads(sys.argv[2])
    template = (ROOT / "agent_run" / "run_agent.py").read_text()
    script = re.sub(r"# CFG-BEGIN\n.*?# CFG-END\n", "CFG = " + json.dumps(cfg, indent=4) + "\n", template,
                    flags=re.S)
    meta = json.loads((ROOT / "agent_run" / "kernel-metadata.json").read_text())
    meta.update(id=f"{USER}/cgloc-{name}", title=f"cgloc {name}")
    out = ROOT / ".build" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "run_agent.py").write_text(script)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    subprocess.run(["kaggle", "kernels", "push", "-p", str(out), "--accelerator", "NvidiaTeslaT4", "-t", "43000"],
                   check=True)


if __name__ == "__main__":
    main()
