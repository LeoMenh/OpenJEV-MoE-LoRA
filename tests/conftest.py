"""Resolve the published snapshot and test-only historical wrappers."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / 'expert_training', ROOT / 'tests/support', ROOT / 'experiments/batch_nll_20261027/source']:
    sys.path.insert(0, str(path))
