"""
Data Loader for NASA/PROMISE software defect prediction datasets.
Normalizes datasets to: complexity, effort, size, doc_density, maintainability, has_bug
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from src.config import Config

AVAILABLE_DATASETS = ["cm1", "jm1", "kc1", "kc2", "pc1"]
FEATURE_COLUMNS = ["complexity", "effort", "size", "doc_density", "maintainability"]


def _load_raw(name: str) -> pd.DataFrame:
    """Loads and normalizes column names from a raw dataset."""
    path = Config.RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Normalize uppercase Halstead columns (pc1 uses E, V, etc.)
    renames = {
        c: c.lower()
        for c in ["E", "V", "N", "L", "D", "I", "B", "T"]
        if c in df.columns
    }
    df = df.rename(columns=renames)

    # Normalize target column (kc2 uses 'problems')
    if "problems" in df.columns:
        df = df.rename(columns={"problems": "defects"})

    df["source_dataset"] = name
    return df


def _to_binary(series: pd.Series) -> pd.Series:
    """Converts target column to binary (0/1)."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        mapping = {
            "true": 1,
            "false": 0,
            "True": 1,
            "False": 0,
            "yes": 1,
            "no": 0,
            "Yes": 1,
            "No": 0,
        }
        return series.map(mapping).fillna(0).astype(int)
    return series.astype(int)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw data to canonical schema."""
    result = pd.DataFrame()

    result["complexity"] = df["v(g)"].astype(float)
    result["effort"] = df["e"].astype(float)
    result["size"] = df["loc"].astype(int)

    # Doc density: comments / (code + comments)
    total = df["lOCode"] + df["lOComment"]
    result["doc_density"] = (
        (df["lOComment"] / total.replace(0, np.nan)).fillna(0).clip(0, 1)
    )

    # Maintainability Index (MI): 171 - 5.2*ln(V) - 0.23*G - 16.2*ln(LOC)
    mi = (
        171
        - 5.2 * np.log(df["v"].clip(1))
        - 0.23 * df["v(g)"].clip(1)
        - 16.2 * np.log(df["loc"].clip(1))
    )
    result["maintainability"] = (mi.clip(0, 171) * 100 / 171).round(2)

    result["has_bug"] = _to_binary(df["defects"])
    result["source_dataset"] = df["source_dataset"]

    return result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Removes metadata rows, invalid values, and duplicates."""
    df = df[df["size"] >= 2].copy()
    for col in FEATURE_COLUMNS:
        df = df[(df[col] >= 0) & (df[col].notna())]
    return df.drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)


def load_dataset(
    name: Optional[str] = None, clean: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Main entry point. Loads one or all datasets unified.

    Args:
        name: Dataset name or None for all
        clean: Apply data cleaning
        verbose: Print progress
    """
    datasets = [name] if name else AVAILABLE_DATASETS

    if name and name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}")

    frames = []
    for ds in datasets:
        try:
            df = _normalize(_load_raw(ds))
            if clean:
                df = _clean(df)
            if verbose:
                print(f"Loaded {ds}: {len(df)} records")
            frames.append(df)
        except FileNotFoundError:
            if verbose:
                print(f"{ds}: not found")

    if not frames:
        raise ValueError("No datasets loaded.")

    result = pd.concat(frames, ignore_index=True)
    if verbose and len(frames) > 1:
        print(
            f"Total: {len(result)} | Bugs: {result['has_bug'].sum()} ({result['has_bug'].mean()*100:.1f}%)"
        )
    return result


def load_datasets_separate(
    datasets: Optional[List[str]] = None, clean: bool = True, verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """Loads datasets as separate DataFrames."""
    datasets = datasets or AVAILABLE_DATASETS
    return {name: load_dataset(name, clean, verbose) for name in datasets}


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) split."""
    return df[FEATURE_COLUMNS], df["has_bug"]


def get_dataset_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Returns descriptive statistics with bug ratio."""
    stats = df[FEATURE_COLUMNS].describe()
    stats.loc["bug_ratio"] = df["has_bug"].mean()
    return stats


def save_processed_dataset(df: pd.DataFrame, filename: str = "unified_metrics.csv"):
    """Saves dataset to processed directory."""
    path = Config.PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    df = load_dataset()
    print(df.head(10))
    save_processed_dataset(df)
