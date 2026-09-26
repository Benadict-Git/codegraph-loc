"""Kaggle T4x2 job: serve a Gemma 4 GGUF with llama.cpp (one server per GPU) and run localization agents."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

CFG = {
    "model_key": "e4b",
    "gpus": 2,
    "slots": 4,
    "ctx_per_slot": 16384,
    "runs": [
        {"name": "pilot_e4b_files", "condition": "files", "limit": 40},
        {"name": "pilot_e4b_graph", "condition": "graph", "limit": 40},
    ],
}
OUT = "/kaggle/working"
json.dump(CFG, open(f"{OUT}/agent_config.json", "w"), indent=2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


sh("pip install -q tree-sitter tree-sitter-python unidiff rank-bm25")
src = glob.glob("/kaggle/input/**/src/pyproject.toml", recursive=True)[0].rsplit("/", 1)[0]
shutil.copytree(src, "/tmp/cgloc", dirs_exist_ok=True)
sh("pip install -q --no-deps -e /tmp/cgloc")

bin_src = glob.glob("/kaggle/input/**/llama-bin/llama-server", recursive=True)[0].rsplit("/", 1)[0]
shutil.copytree(bin_src, "/tmp/llama-bin", dirs_exist_ok=True)
sh("chmod +x /tmp/llama-bin/*")
env = dict(os.environ, LD_LIBRARY_PATH="/tmp/llama-bin:" + os.environ.get("LD_LIBRARY_PATH", ""))

gguf = sorted(g for g in glob.glob("/kaggle/input/**/*.gguf", recursive=True)
              if CFG["model_key"] in os.path.basename(g).lower() and "mmproj" not in g.lower())[0]
log(f"model: {gguf}")
servers, urls = [], []
for gpu in range(CFG.get("gpus", 2)):
    port = 8080 + gpu
    logf = open(f"{OUT}/server{gpu}.log", "w")
    servers.append(subprocess.Popen(
        ["/tmp/llama-bin/llama-server", "-m", gguf, "-ngl", "999", "-c", str(CFG["ctx_per_slot"] * CFG["slots"]),
         "-np", str(CFG["slots"]), "--jinja", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
         "--port", str(port)], stdout=logf, stderr=subprocess.STDOUT,
        env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))))
    urls.append(f"http://127.0.0.1:{port}")
for u, s in zip(urls, servers):
    t = time.time()
    while True:
        try:
            with urllib.request.urlopen(u + "/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if s.poll() is not None or time.time() - t > 900:
            sys.exit(f"server {u} failed; see server logs")
        time.sleep(3)
log(f"servers ready: {urls}")

try:
    for run in CFG["runs"]:
        out = f"{OUT}/{run['name']}.jsonl"
        cmd = [sys.executable, "-m", "cgloc.agent.run", "--dataset", run.get("dataset", "lite"),
               "--split", run.get("split", "test"), "--condition", run["condition"],
               "--base-urls", ",".join(urls), "--workers", str(CFG["slots"] * len(urls)),
               "--budget", str(run.get("budget", 12)), "--max-tokens", str(run.get("max_tokens", 1024)),
               "--hint", run.get("hint", "bm25"), "--cache-dir", "/tmp/repos", "--out", out]
        if run.get("limit"):
            cmd += ["--limit", str(run["limit"])]
        if run.get("thinking"):
            cmd.append("--thinking")
        log(" ".join(cmd))
        subprocess.run(cmd, check=False)
finally:
    for s in servers:
        s.terminate()
log("done")
