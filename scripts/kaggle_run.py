"""Run a repo command on a Kaggle GPU and fetch its outputs (T16).

    py -3 scripts/kaggle_run.py --slug ria-detect-whatsup \
        --cmd "python scripts/download_whatsup.py && python scripts/detect_boxes.py --config configs/detect_whatsup.json" \
        --outputs data/boxes results

Steps: package the committed code (git archive HEAD) as a private Kaggle dataset
`<user>/ria-code`, push a private GPU script kernel that unpacks it in /tmp, runs
--cmd, and copies --outputs to /kaggle/working/out; then poll and download the
outputs back into the same paths locally (files are overwritten, others kept).

Needs Kaggle credentials (`py -3 -m kaggle auth login`) and a phone-verified
account (required for GPU and internet in kernels).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / ".kaggle_staging"
CODE_DATASET = "ria-code"

KERNEL_TEMPLATE = '''\
import glob, os, shutil, subprocess, sys, tarfile
CMD = {cmd!r}
OUTPUTS = {outputs!r}
os.environ["RIA_GIT_COMMIT"] = {commit!r}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", {pins}], check=True)
work = "/tmp/ria"
src = glob.glob("/kaggle/input/**/ria-code.tar.gz", recursive=True)
if src:
    os.makedirs(work, exist_ok=True)
    with tarfile.open(src[0]) as t:
        t.extractall(work)
else:  # Kaggle may have extracted the archive on upload
    roots = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)]
    assert roots, "code dataset not mounted"
    shutil.copytree(roots[0], work)
os.chdir(work)
# Output dirs that do not exist yet are symlinked into /kaggle/working, so results are
# saved as they are written and survive a session timeout; existing paths are copied at the end.
live = []
for p in OUTPUTS:
    if not os.path.exists(p):
        dst = os.path.join("/kaggle/working/out", p)
        os.makedirs(dst, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        os.symlink(dst, p)
        live.append(p)
print("running:", CMD, flush=True)
rc = subprocess.run(CMD, shell=True).returncode
for p in OUTPUTS:
    if p not in live and os.path.exists(p):
        dst = os.path.join("/kaggle/working/out", p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, dst)
print("exit code", rc, flush=True)
sys.exit(rc)
'''


def kaggle(*args, check=True, capture=True) -> str:
    # UTF-8 everywhere: the CLI echoes kernel logs, which crash the Windows console codepage.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=capture, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def username() -> str:
    out = kaggle("config", "view", check=False)
    for line in out.splitlines():
        if "username" in line.lower() and ":" in line:
            name = line.split(":", 1)[1].strip()
            if name and name.lower() != "none":
                return name
    raise SystemExit("could not read the Kaggle username; pass --user or run `py -3 -m kaggle auth login`")


def upload_code(user: str) -> None:
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: uncommitted changes are NOT uploaded (git archive HEAD)")
    d = STAGING / "code"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(d / "ria-code.tar.gz"), "HEAD"], cwd=REPO_ROOT, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"title": CODE_DATASET, "id": f"{user}/{CODE_DATASET}", "licenses": [{"name": "other"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    exists = kaggle("datasets", "status", f"{user}/{CODE_DATASET}", check=False)
    if "ready" in exists.lower() or "pending" in exists.lower():
        print(kaggle("datasets", "version", "-p", str(d), "-m", f"code at {head}").strip())
    else:
        print(kaggle("datasets", "create", "-p", str(d)).strip())          # private by default
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", f"{user}/{CODE_DATASET}", check=False).lower():
            print(f"code dataset ready ({head})")
            return
        time.sleep(10)
    raise SystemExit("code dataset not ready after 10 minutes")


def push_kernel(user: str, slug: str, cmd: str, outputs: list[str], pins: list[str], accelerator: str | None,
                datasets: list[str] = (), gpu: bool = True) -> str:
    d = STAGING / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    pin_args = ", ".join(repr(p) for p in pins)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    (d / "kernel.py").write_text(KERNEL_TEMPLATE.format(cmd=cmd, outputs=outputs, pins=pin_args, commit=commit),
                                 encoding="utf-8")
    kid = f"{user}/{slug}"
    meta = {"id": kid, "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [f"{user}/{CODE_DATASET}"] + [d if "/" in d else f"{user}/{d}" for d in datasets], "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    args = ["kernels", "push", "-p", str(d)] + (["--accelerator", accelerator] if accelerator and gpu else [])
    print(kaggle(*args).strip())
    return kid


def kernel_status(kid: str) -> str | None:
    """Kaggle's status word (running, queued, complete, error, ...), or None if the check failed.

    Network failures must not be mistaken for a failed kernel, so only the status
    value itself is parsed (their messages contain words like "Error").
    """
    import re
    out = kaggle("kernels", "status", kid, check=False)
    m = re.search(r'has status "(?:kernelworkerstatus\.)?([a-z_]+)"', out, re.IGNORECASE)
    return m.group(1).lower() if m else None


def wait(kid: str, poll: int) -> str:
    t0 = time.time()
    last = None
    while True:
        state = kernel_status(kid)
        if state in ("complete", "error", "cancel_acknowledged", "cancel_requested", "cancelacknowledged"):
            print(f"kernel {state} after {(time.time() - t0) / 60:.1f} min", flush=True)
            return state
        if state != last:
            print(f"  {(time.time() - t0) / 60:.1f} min: {state or 'status unavailable (network?), retrying'}", flush=True)
            last = state
        time.sleep(poll)


def fetch(kid: str, slug: str) -> None:
    d = STAGING / f"{slug}-output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    kaggle("kernels", "output", kid, "-p", str(d))
    out = d / "out"
    log = next(d.glob("*.log"), None)
    if log:
        print("--- kernel log tail ---\n" + "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    if not out.exists():
        print("no outputs were produced")
        return
    for f in out.rglob("*"):
        if f.is_file():
            dst = REPO_ROOT / f.relative_to(out)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            print(f"  fetched {dst.relative_to(REPO_ROOT)}")


def main():
    # CLI replies include progress-bar glyphs the Windows console codepage cannot print.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="kernel name, e.g. ria-detect-whatsup")
    ap.add_argument("--cmd", required=True, help="shell command run from the repo root on Kaggle")
    ap.add_argument("--outputs", nargs="*", default=["results"], help="repo paths to bring back")
    ap.add_argument("--pins", nargs="*", default=["transformers==4.52.4", "gdown", "sentencepiece"])
    ap.add_argument("--datasets", nargs="*", default=[], help="extra Kaggle datasets to attach, e.g. whatsup-controlled")
    ap.add_argument("--cpu", action="store_true", help="CPU-only kernel (no GPU quota used), e.g. for FIxLIP")
    ap.add_argument("--accelerator", default=None, help="e.g. NvidiaTeslaT4 (default: Kaggle's default GPU)")
    ap.add_argument("--user", default=None)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--skip-upload", action="store_true", help="reuse the last uploaded code")
    ap.add_argument("--fetch-only", action="store_true", help="only download outputs of the last run")
    ap.add_argument("--wait-only", action="store_true", help="wait for a running kernel, then fetch its outputs")
    args = ap.parse_args()

    user = args.user or username()
    kid = f"{user}/{args.slug}"
    if args.wait_only:
        print(f"kernel {wait(kid, args.poll)}")
    elif not args.fetch_only:
        if not args.skip_upload:
            upload_code(user)
        push_kernel(user, args.slug, args.cmd, args.outputs, args.pins, args.accelerator, args.datasets,
                    gpu=not args.cpu)
        if wait(kid, args.poll) != "complete":
            fetch(kid, args.slug)
            raise SystemExit(f"kernel did not complete; see https://www.kaggle.com/code/{kid}")
    fetch(kid, args.slug)


if __name__ == "__main__":
    main()
