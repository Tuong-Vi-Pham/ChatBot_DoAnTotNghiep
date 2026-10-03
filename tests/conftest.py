import sys
import pathlib

# Ensure the repository root (parent of the tests directory) is in PYTHONPATH
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
