"""
Workaround for a packaging bug in the upstream ibug-group repos.

Both https://github.com/ibug-group/face_detection and
https://github.com/ibug-group/face_alignment declare only their top-level
package in setup.py (e.g. `'packages': ['ibug.face_detection']`), omitting
their actual subpackages (s3fd, retina_face, fan, utils, ...). A plain
`pip install git+https://...` therefore installs an ibug.face_detection /
ibug.face_alignment that is missing everything except __init__.py, and
`from ibug.face_detection import RetinaFacePredictor` fails with
`ModuleNotFoundError: No module named 'ibug.face_detection.s3fd'`.

This script clones both repos to a temp dir and copies their missing
subpackages (weights included) into the already-installed site-packages
location. Run it once after `pip install -r requirements.txt`.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOS = {
    "https://github.com/ibug-group/face_detection.git": {
        "package": "ibug.face_detection",
        "subdirs": ["s3fd", "retina_face", "utils"],
    },
    "https://github.com/ibug-group/face_alignment.git": {
        "package": "ibug.face_alignment",
        "subdirs": ["fan"],
    },
}


def install_dir_for(package: str) -> Path:
    module_name = package.split(".")[0]
    module = __import__(module_name)
    return Path(module.__path__[0]) / package.split(".", 1)[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for url, info in REPOS.items():
            dest = install_dir_for(info["package"])
            if not dest.is_dir():
                sys.exit(f"{info['package']} is not installed; run pip install -r requirements.txt first")

            clone_dir = Path(tmp) / info["package"]
            subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(clone_dir)], check=True)

            pkg_subpath = info["package"].replace(".", "/")
            for subdir in info["subdirs"]:
                src = clone_dir / pkg_subpath / subdir
                if not src.is_dir():
                    sys.exit(f"expected {src} in {url}, but it was not found")
                shutil.copytree(src, dest / subdir, dirs_exist_ok=True)
                print(f"copied {subdir} -> {dest / subdir}")


if __name__ == "__main__":
    main()
