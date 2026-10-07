"""Check that every package this folder needs is installed, and new enough.

    python check_install.py

Prints one line per package: what it is for, the version found, and OK or
what is wrong. Ends with the command that installs anything missing. Exit
status 0 when everything is in place, 1 otherwise. verify.py runs this first.
"""
from importlib import import_module
import sys

# (import name, pip name, lowest version, what it is for)
REQUIRED = [
    ("numpy", "numpy", "1.23", "arrays, everywhere"),
    ("scipy", "scipy", "1.9", "filters and spectra"),
    ("cv2", "opencv-python-headless", "4.6", "blob detection and video frames"),
    ("numba", "numba", "0.57", "the fast noise filter; without it results differ slightly"),
    ("pandas", "pandas", "1.5", "output tables"),
    ("matplotlib", "matplotlib", "3.5", "figures"),
    ("imageio_ffmpeg", "imageio-ffmpeg", "0.4", "writing MP4 clips"),
    ("faery", "faery", "0.7", "reading .raw, .raw.kai, .dat, .es and .aedat files"),
    ("h5py", "h5py", "3.0", "reading .h5 / .hdf5 files"),
]
PYTHON_MIN = (3, 9)


def version_tuple(text):
    """'1.26.4' -> (1, 26, 4); stops at the first part that is not a number."""
    out = []
    for part in str(text).split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
        if digits != part:
            break
    return tuple(out)


def check():
    """Return a list of (pip name, problem) for everything missing or too old."""
    problems = []
    ok_py = sys.version_info[:2] >= PYTHON_MIN
    print("{:<24} {:<14} {}".format("Python", sys.version.split()[0],
                                    "OK" if ok_py else "needs {}.{} or newer".format(*PYTHON_MIN)))
    if not ok_py:
        problems.append(("python", "too old"))
    for module, pip_name, lowest, purpose in REQUIRED:
        try:
            mod = import_module(module)
            found = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None) or "?"
            if found != "?" and version_tuple(found) < version_tuple(lowest):
                status = "too old, needs {} or newer".format(lowest)
                problems.append((pip_name, status))
            else:
                status = "OK"
        except Exception as exc:          # missing, or installed but broken
            found = "-"
            status = "MISSING" if isinstance(exc, ImportError) else "BROKEN: {}".format(exc)
            problems.append((pip_name, status))
        print("{:<24} {:<14} {:<34} {}".format(pip_name, str(found), status, purpose))
    return problems


def main() -> int:
    problems = check()
    if problems:
        names = " ".join(sorted({name for name, _ in problems if name != "python"}))
        print("\n{} problem(s). Install with:".format(len(problems)))
        print("    python -m pip install -r requirements.txt" + ("" if not names else
              "\n  or just these:\n    python -m pip install --upgrade " + names))
        return 1
    print("\nEverything needed is installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
