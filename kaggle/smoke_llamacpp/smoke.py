"""Kaggle T4 smoke test: build llama.cpp (CUDA sm75), load Gemma 4 GGUF models, measure generation speed."""
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import time
import urllib.request

OUT = "/kaggle/working"
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, **kw):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.returncode:
        log(f"FAILED ({r.returncode})\n--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-4000:]}")
        raise subprocess.CalledProcessError(r.returncode, cmd)
    return r.stdout


def save():
    with open(f"{OUT}/smoke_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)


RESULTS["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
log(RESULTS["nvidia_smi"])
RESULTS["ggufs"] = sorted(glob.glob("/kaggle/input/**/*.gguf", recursive=True))
log(f"gguf files: {RESULTS['ggufs']}")
save()

nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)), None)
RESULTS["nvcc"] = nvcc
if nvcc:
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ["PATH"]
    os.environ["CUDACXX"] = nvcc
    os.environ["CUDA_HOME"] = cuda_home
    RESULTS["nvcc_version"] = sh(f"{nvcc} --version")
RESULTS["cmake_version"] = sh("cmake --version")
log(f"nvcc={nvcc}\n{RESULTS['cmake_version']}")
save()

t0 = time.time()
sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
RESULTS["llama_cpp_commit"] = sh("git -C /tmp/llama.cpp rev-parse HEAD").strip()
drivers = [p for pat in ("/usr/local/cuda*/lib64/stubs/libcuda.so", "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
                         "/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/local/nvidia/lib64/libcuda.so*",
                         "/usr/lib64/libcuda.so*") for p in sorted(glob.glob(pat))]
RESULTS["libcuda_candidates"] = drivers
log(f"libcuda candidates: {drivers}")
cuda_flag = f"-DCMAKE_CUDA_COMPILER={nvcc} " if nvcc else ""
if drivers:
    cuda_flag += f"-DCUDA_cuda_driver_LIBRARY={drivers[0]} "
sh(f"cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 {cuda_flag}"
   "-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release")
sh(f"cmake --build /tmp/llama.cpp/build -j{os.cpu_count()} --target llama-server llama-cli")
RESULTS["build_secs"] = round(time.time() - t0, 1)
log(f"built in {RESULTS['build_secs']}s")
os.makedirs(f"{OUT}/llama-bin", exist_ok=True)
for f in glob.glob("/tmp/llama.cpp/build/bin/*"):
    shutil.copy2(f, f"{OUT}/llama-bin/")
save()


def chat(port, prompt, max_tokens=256):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return resp, time.time() - t


PROMPT = ("You are localizing a bug. Issue: `QuerySet.update()` ignores `F()` expressions on related fields. "
          "List the three Django functions most likely to need changes, as `path::Class.method`, then explain why.")

for gguf in RESULTS["ggufs"]:
    name = os.path.basename(gguf)
    if "mmproj" in name.lower():
        continue
    entry = {"file": gguf, "size_gb": round(os.path.getsize(gguf) / 1e9, 2)}
    RESULTS.setdefault("models", {})[name] = entry
    port = 8080
    logf = open(f"{OUT}/server_{name}.log", "w")
    srv = subprocess.Popen([f"{OUT}/llama-bin/llama-server", "-m", gguf, "-ngl", "999", "-c", "32768", "-np", "4",
                            "--jinja", "--port", str(port), "-fa", "on"], stdout=logf, stderr=subprocess.STDOUT)
    try:
        t = time.time()
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if srv.poll() is not None or time.time() - t > 900:
                raise RuntimeError("server failed to start")
            time.sleep(3)
        entry["load_secs"] = round(time.time() - t, 1)
        resp, secs = chat(port, PROMPT)
        n = resp["usage"]["completion_tokens"]
        entry["single"] = {"tokens": n, "secs": round(secs, 2), "tok_per_s": round(n / secs, 1)}
        entry["sample"] = resp["choices"][0]["message"].get("content", "")[:1500]
        with cf.ThreadPoolExecutor(4) as ex:
            t = time.time()
            outs = list(ex.map(lambda _: chat(port, PROMPT), range(4)))
            wall = time.time() - t
        total = sum(o[0]["usage"]["completion_tokens"] for o in outs)
        entry["parallel4"] = {"tokens": total, "secs": round(wall, 2), "tok_per_s": round(total / wall, 1)}
        entry["vram"] = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
        log(f"{name}: {entry}")
    except Exception as e:
        entry["error"] = repr(e)
        log(f"{name}: ERROR {e!r}")
    finally:
        srv.terminate()
        srv.wait(timeout=60)
        logf.close()
        save()

RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done")
