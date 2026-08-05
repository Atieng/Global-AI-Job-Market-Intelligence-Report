"""
data_prep.py
============
CRISP-DM stages: Data Understanding + Data Preparation

Reusable loading, profiling, and cleaning functions. Works on any tabular
dataset as long as config.py points to the right column names.
"""

import pandas as pd
import numpy as np
from config import DatasetConfig


def load_and_merge_data(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Load multiple dataset files (cfg.raw_data_paths) and concatenate them,
    tagging each row with a 'source' column (cfg.source_labels) so combined
    analysis or source-vs-source comparison is possible later.
    """
    if len(cfg.raw_data_paths) != len(cfg.source_labels):
        raise ValueError("raw_data_paths and source_labels must be the same length")

    frames = []
    for path, label in zip(cfg.raw_data_paths, cfg.source_labels):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        df[cfg.source_col] = label
        print(f"[load_and_merge_data] Loaded {df.shape[0]} rows from {path} (source={label})")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[load_and_merge_data] Combined shape: {combined.shape}")
    return combined


def decode_abbreviations(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Map coded columns (experience_level, employment_type) to readable labels."""
    df = df.copy()
    if cfg.experience_col in df.columns and cfg.experience_level_map:
        df[cfg.experience_col] = df[cfg.experience_col].map(cfg.experience_level_map).fillna(df[cfg.experience_col])
    if cfg.employment_type_col in df.columns and cfg.employment_type_map:
        df[cfg.employment_type_col] = df[cfg.employment_type_col].map(cfg.employment_type_map).fillna(df[cfg.employment_type_col])
    return df


def load_data(cfg: DatasetConfig) -> pd.DataFrame:
    """Load raw data from CSV/Excel based on file extension."""
    path = cfg.raw_data_path
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    print(f"[load_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quick data understanding summary: dtypes, missing %, unique counts."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    print("[profile_data] Summary:\n", summary)
    return summary


def clean_data(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Generic cleaning steps:
    - drop exact duplicate rows
    - strip whitespace from string columns
    - coerce salary column to numeric
    - drop rows missing the regression/classification targets
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean_data] Dropped {before - len(df)} duplicate rows")

    # Strip whitespace on object/string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Coerce salary to numeric, removing common currency symbols/commas
    if cfg.salary_col in df.columns:
        df[cfg.salary_col] = (
            df[cfg.salary_col]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df[cfg.salary_col] = pd.to_numeric(df[cfg.salary_col], errors="coerce")

    # Drop rows with missing critical targets
    critical_cols = [c for c in [cfg.regression_target, cfg.classification_target] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"[clean_data] Dropped {before - len(df)} rows missing target columns {critical_cols}")

    df = df.reset_index(drop=True)
    return df


def standardize_remote_column(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Normalize the remote/on-site column into a consistent category:
    'Remote', 'Hybrid', 'On-site' — handles either numeric (0/50/100)
    or text-based source columns.
    """
    df = df.copy()
    col = cfg.remote_col
    if col not in df.columns:
        return df

    def map_value(v):
        if isinstance(v, str):
            v_lower = v.lower()
            if "remote" in v_lower:
                return "Remote"
            if "hybrid" in v_lower:
                return "Hybrid"
            if "on" in v_lower:  # on-site / onsite
                return "On-site"
            return v
        try:
            v_num = float(v)
            if v_num >= 80:
                return "Remote"
            if v_num <= 20:
                return "On-site"
            return "Hybrid"
        except (ValueError, TypeError):
            return v

    df[col] = df[col].apply(map_value)
    return df


def run_data_prep(cfg: DatasetConfig) -> pd.DataFrame:
    """Convenience wrapper: load -> profile -> clean -> standardize (single file)."""
    df = load_data(cfg)
    profile_data(df)
    df = clean_data(df, cfg)
    df = standardize_remote_column(df, cfg)
    return df


def run_data_prep_merged(cfg: DatasetConfig) -> pd.DataFrame:
    """Convenience wrapper for the multi-file case: merge -> decode -> profile -> clean -> standardize."""
    df = load_and_merge_data(cfg)
    df = decode_abbreviations(df, cfg)
    profile_data(df)
    df = clean_data(df, cfg)
    df = standardize_remote_column(df, cfg)
    return df


if __name__ == "__main__":
    from config import CONFIG
    df = run_data_prep(CONFIG)
    print(df.head())
