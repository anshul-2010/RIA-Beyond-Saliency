"""Download and extract the What's Up controlled image sets (subsets A and B).

    py -3 scripts/download_whatsup.py [--root data/whatsup]

Files come from the authors' Google Drive (IDs from github.com/amitakamath/whatsup_vlms).
Archive members are checked before extraction; anything outside the expected
folders is refused.
"""

import argparse
import re
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "controlled_images_dataset.json": "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
    "controlled_clevr_dataset.json": "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z",
    "controlled_images.tar.gz": "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
    "controlled_clevr.tar.gz": "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG",
}
SAFE_MEMBER = re.compile(r"^(\./)?controlled_(images|clevr)(/[A-Za-z0-9_.()-]+)?/?$")


def safe_extract(archive: Path, dest: Path) -> int:
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        bad = [m.name for m in members if not SAFE_MEMBER.match(m.name) or not (m.isfile() or m.isdir())]
        if bad:
            raise RuntimeError(f"refusing to extract {archive.name}: unexpected members {bad[:5]}")
        tar.extractall(dest, members=members, filter="data")   # members already checked above
    return len(members)


def from_mirror(src: Path, root: Path) -> None:
    """Fill `root` from a mirror holding the JSONs and either the archives or unpacked folders."""
    import shutil
    for name in FILES:
        found = sorted(src.rglob(name))
        folder = name.replace(".tar.gz", "")
        if name.endswith(".json"):
            if not found:
                raise FileNotFoundError(f"{name} not found under {src}")
            shutil.copy2(found[0], root / name)
        elif (root / folder).exists():
            continue
        elif found:
            print(f"extracting {name}: {safe_extract(found[0], root)} members", flush=True)
        else:                                   # mirror already unpacked the archive
            dirs = [d for d in src.rglob(folder) if d.is_dir() and any(d.glob("*.jpeg"))]
            if not dirs:
                raise FileNotFoundError(f"neither {name} nor {folder}/ found under {src}")
            shutil.copytree(dirs[0], root / folder)
            print(f"copied {folder}/ ({sum(1 for _ in (root / folder).glob('*.jpeg'))} images)", flush=True)
    print(f"What's Up ready in {root} (from {src})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "whatsup")
    ap.add_argument("--from-dir", type=Path, default=None,
                    help="copy from a local mirror (e.g. an attached Kaggle dataset) instead of Google Drive")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        return from_mirror(args.from_dir, args.root)
    import gdown

    for name, file_id in FILES.items():
        out = args.root / name
        if not out.exists():
            print(f"downloading {name}", flush=True)
            gdown.download(id=file_id, output=str(out), quiet=True)
        if name.endswith(".tar.gz") and not (args.root / name.replace(".tar.gz", "")).exists():
            print(f"extracting {name}: {safe_extract(out, args.root)} members", flush=True)
    print(f"What's Up ready in {args.root}")


if __name__ == "__main__":
    sys.exit(main())
