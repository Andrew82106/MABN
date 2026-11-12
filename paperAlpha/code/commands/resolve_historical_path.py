"""Print a historical artifact's current location without modifying files."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/src"))
from paperalpha_runtime.workspace import resolve_historical_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(resolve_historical_path(args.path))
