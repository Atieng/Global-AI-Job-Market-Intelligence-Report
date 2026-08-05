"""
config.py
=========
Single place to adapt this pipeline to a NEW dataset.
Change the values below to match your dataset's column names and you can
reuse every other .py file unchanged.

CRISP-DM stage: Business Understanding / Data Understanding (setup)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    # ---- File location(s) ----
    # Single-file mode (used by data_prep.load_data)
    raw_data_path: str = "data/ai_job_dataset.csv"
    # Multi-file mode (used by data_prep.load_and_merge_data) — merges all
    # files listed here, tagging each row with a 'source' column so you can
    # analyze combined data or compare the sources.
    raw_data_paths: List[str] = field(default_factory=lambda: [
        "data/ai_job_dataset.csv",
        "data/ai_job_dataset1.csv",
    ])
    source_labels: List[str] = field(default_factory=lambda: ["dataset_A", "dataset_B"])
    source_col: str = "source"

    output_dir: str = "outputs"
    model_dir: str = "models"

    # ---- Core columns (match ai_job_dataset.csv / ai_job_dataset1.csv) ----
    salary_col: str = "salary_usd"
    country_col: str = "company_location"
    experience_col: str = "experience_level"           # coded EN/MI/SE/EX
    remote_col: str = "remote_ratio"                    # 0 / 50 / 100
    skills_col: str = "required_skills"                 # comma-separated free text
    job_title_col: str = "job_title"

    # ---- Extra columns present in this dataset ----
    employment_type_col: str = "employment_type"        # coded FT/PT/CT/FL
    company_size_col: str = "company_size"               # S/M/L
    education_col: str = "education_required"
    industry_col: str = "industry"
    years_experience_col: str = "years_experience"
    posting_date_col: str = "posting_date"
    application_deadline_col: str = "application_deadline"

    # Mappings to decode abbreviations into readable labels
    experience_level_map: dict = field(default_factory=lambda: {
        "EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive",
    })
    employment_type_map: dict = field(default_factory=lambda: {
        "FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance",
    })

    # ---- Target definitions ----
    regression_target: str = "salary_usd"
    classification_target: str = "experience_level"     # what we predict as "job level"

    # ---- Columns to drop before modeling (IDs, free text not used as features) ----
    drop_for_model: List[str] = field(default_factory=lambda: [
        "job_id", "job_title", "required_skills", "company_name",
        "posting_date", "application_deadline", "salary_currency", "salary_local",
    ])

    # ---- Skill list separator, if skills are stored as one string per row ----
    skills_separator: str = ","

    # ---- Top-N skills to report ----
    top_n_skills: int = 20

    # ---- Modeling ----
    test_size: float = 0.2
    random_state: int = 42


# Default instance — import this in scripts, or build your own DatasetConfig(...)
CONFIG = DatasetConfig()
