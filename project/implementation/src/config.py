from pathlib import Path


class Config:

    # General settings
    PROJECT_NAME = "Repo-Aura"
    SEED = 42

    # Dir definition
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"

    # Automatic dirs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Unic path to dataset
    DATASET_PATH = RAW_DIR / "dataset_metrics.csv"

    # Thresholds
    THRESHOLD_COMPLEXITY = 15
    THRESHOLD_COVERAGE = 40
