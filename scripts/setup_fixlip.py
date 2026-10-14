"""Clone the official FIxLIP code at a pinned commit and install its dependency.

    py -3 scripts/setup_fixlip.py

FIxLIP (Baniecki et al., NeurIPS 2025, MIT licence) is not on pip. It goes to
third_party/fixlip (git-ignored); ria.baselines.FixlipBaseline imports it from there.
"""

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/hbaniecki/fixlip.git"
COMMIT = "df830df51cda1a33b2b918e515b82bb8fc17e338"
SHAPIQ = ["shapiq==1.3.0", "overrides"]   # shapiq pinned in FIxLIP's env.yml; overrides is an undeclared import
DEST = Path(__file__).resolve().parents[1] / "third_party" / "fixlip"


def run(*cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", REPO, str(DEST))
    run("git", "fetch", "--quiet", "origin", COMMIT, cwd=DEST)
    run("git", "checkout", "--quiet", COMMIT, cwd=DEST)
    run(sys.executable, "-m", "pip", "install", "-q", *SHAPIQ)
    print(f"FIxLIP {COMMIT[:7]} ready in {DEST}")


if __name__ == "__main__":
    main()
