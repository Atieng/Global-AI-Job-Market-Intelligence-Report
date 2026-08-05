"""
eda.py
======
CRISP-DM stage: Data Understanding (exploratory analysis)

All functions return DataFrames/Series so they can be reused in a notebook,
a script, or fed straight into a report generator.
"""

import pandas as pd
from collections import Counter
from config import DatasetConfig


def top_skills(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Return the top-N most frequently requested skills.
    Handles a skills column stored as a delimited string per row
    (e.g. "Python, SQL, TensorFlow").
    """
    if cfg.skills_col not in df.columns:
        raise KeyError(f"Skills column '{cfg.skills_col}' not found in dataframe")

    all_skills = []
    for entry in df[cfg.skills_col].dropna():
        parts = [s.strip() for s in str(entry).split(cfg.skills_separator) if s.strip()]
        all_skills.extend(parts)

    counts = Counter(all_skills)
    top = counts.most_common(cfg.top_n_skills)
    result = pd.DataFrame(top, columns=["skill", "count"])
    result["pct_of_postings"] = (result["count"] / len(df) * 100).round(2)
    return result


def salary_by_group(df: pd.DataFrame, cfg: DatasetConfig, group_col: str) -> pd.DataFrame:
    """Average and median salary grouped by an arbitrary column (country, experience, remote status)."""
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe")

    result = (
        df.groupby(group_col)[cfg.salary_col]
        .agg(mean_salary="mean", median_salary="median", count="count")
        .round(0)
        .sort_values("mean_salary", ascending=False)
        .reset_index()
    )
    return result


def salary_by_country(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    return salary_by_group(df, cfg, cfg.country_col)


def salary_by_experience(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    return salary_by_group(df, cfg, cfg.experience_col)


def salary_by_remote_status(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    return salary_by_group(df, cfg, cfg.remote_col)


def remote_distribution(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Percentage distribution of remote / hybrid / on-site roles."""
    counts = df[cfg.remote_col].value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    result = pd.DataFrame({"count": counts, "pct": pct})
    return result.reset_index().rename(columns={"index": cfg.remote_col})


def experience_salary_correlation(df: pd.DataFrame, cfg: DatasetConfig) -> float:
    """
    Correlation between experience level and salary.
    Experience level is ordinal (categorical), so it is label-encoded
    in a sensible order before computing Pearson correlation.
    """
    order_guess = ["Entry", "EN", "Junior", "Mid", "MI", "Mid-level", "Senior", "SE", "Lead", "Executive", "EX", "Principal"]
    unique_levels = df[cfg.experience_col].dropna().unique().tolist()

    # Order known levels by the guessed hierarchy; unknown levels appended at the end
    ordered = [lvl for lvl in order_guess if lvl in unique_levels]
    ordered += [lvl for lvl in unique_levels if lvl not in ordered]
    mapping = {lvl: i for i, lvl in enumerate(ordered)}

    encoded = df[cfg.experience_col].map(mapping)
    corr = encoded.corr(df[cfg.salary_col])
    print(f"[experience_salary_correlation] Order used: {ordered}")
    return corr


def full_correlation_matrix(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Correlation matrix across all numeric columns, salary included."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr()


def run_eda(df: pd.DataFrame, cfg: DatasetConfig) -> dict:
    """Run all EDA steps and return results as a dict of DataFrames/values."""
    results = {}
    results["top_skills"] = top_skills(df, cfg)
    results["salary_by_country"] = salary_by_country(df, cfg)
    results["salary_by_experience"] = salary_by_experience(df, cfg)
    results["salary_by_remote_status"] = salary_by_remote_status(df, cfg)
    results["remote_distribution"] = remote_distribution(df, cfg)
    results["experience_salary_correlation"] = experience_salary_correlation(df, cfg)
    results["correlation_matrix"] = full_correlation_matrix(df, cfg)
    return results


if __name__ == "__main__":
    from config import CONFIG
    from data_prep import run_data_prep

    df = run_data_prep(CONFIG)
    results = run_eda(df, CONFIG)
    for name, val in results.items():
        print(f"\n=== {name} ===")
        print(val)
