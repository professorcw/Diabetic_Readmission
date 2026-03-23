# paths.py  —  place this at project_root/
from pathlib import Path

def get_project_root() -> Path:
    """Walk up from the calling file until we find paths.py (i.e., the project root)."""
    here = Path(__file__).resolve().parent
    return here

ROOT      = get_project_root()
DATA_DIR  = ROOT / 'Data'
FIG_DIR   = ROOT / 'figures'
MODEL_DIR = ROOT / 'models'
ART_DIR   = ROOT / 'artifacts'

# Ensure output folders exist when this module is imported
for _dir in [FIG_DIR, MODEL_DIR, ART_DIR, DATA_DIR]:
    _dir.mkdir(exist_ok=True)

# Convenience: key files
DIABETES_CSV = DATA_DIR / 'diabetic_data.csv'
IDS_CSV      = DATA_DIR / 'IDs_mapping.csv'