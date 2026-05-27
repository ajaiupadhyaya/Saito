"""Generate bundled CSV datasets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_preset

DATA_DIR = ROOT / "datasets"
DATA_DIR.mkdir(exist_ok=True)

generate_preset("crash_correlation", n_obs=500).to_csv(DATA_DIR / "corr_noise_50x50.csv")
generate_preset("ou_spread", n_obs=600).to_csv(DATA_DIR / "pair_spread_ou.csv")
generate_preset("fat_t_student", n_obs=800).to_csv(DATA_DIR / "tail_events.csv")
print("Wrote datasets to", DATA_DIR)
