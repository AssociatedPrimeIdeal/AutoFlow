"""Generate the retained 24 analytic phantom inputs."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autoflow.algorithms.regression_phantoms import generate_regression_matrix
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/phantoms')
    args = parser.parse_args()
    rows = generate_regression_matrix(args.output)
    print(f'Generated {len(rows)} inputs: {args.output}')
if __name__ == '__main__':
    main()
