"""Generate the multi-resolution analytic flow benchmark, without overwriting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autoflow.algorithms.analytic_phantoms import generate_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'analytic_phantoms')
    args = parser.parse_args()
    generate_matrix(args.output)
