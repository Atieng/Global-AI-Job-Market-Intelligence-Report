# Using the pipeline from `index.ipynb`

Copy `config.py`, `data_prep.py`, `eda.py`, `modeling.py` into the same folder
as `index.ipynb` (e.g. next to your `Data/` folder), then paste these cells.

## Cell 1 — Setup
```python
import sys
sys.path.append(".")  # folder containing the .py files, adjust if needed

from config import CONFIG
from data_prep import run_data_prep_merged
from eda import run_eda
from modeling import build_regression_model, build_classification_model

# point at your two files
CONFIG.raw_data_paths = ["Data/ai_job_dataset.csv", "Data/ai_job_dataset1.csv"]
CONFIG.source_labels = ["dataset_A", "dataset_B"]
```

## Cell 2 — Load, clean, merge (adds a `source` column, decodes EN/MI/SE/EX etc.)
```python
df = run_data_prep_merged(CONFIG)
df.head()
```

## Cell 3 — EDA: top 20 skills
```python
results = run_eda(df, CONFIG)
results["top_skills"]
```

## Cell 4 — Salary by country / experience / remote status
```python
results["salary_by_country"]
```
```python
results["salary_by_experience"]
```
```python
results["salary_by_remote_status"]
```

## Cell 5 — Remote/hybrid/on-site distribution
```python
results["remote_distribution"]
```

## Cell 6 — Experience-salary correlation
```python
print("Correlation:", results["experience_salary_correlation"])
results["correlation_matrix"]
```

## Cell 7 — Compare dataset_A vs dataset_B directly
```python
df.groupby("source")["salary_usd"].agg(["mean", "median", "count"])
```
```python
import pandas as pd
pd.crosstab(df["source"], df["experience_level"], normalize="index").round(3)
```

## Cell 8 — Regression model (predict salary)
```python
reg_results = build_regression_model(df, CONFIG)
reg_results["metrics"]
```

## Cell 9 — Classification model (predict job level)
```python
clf_results = build_classification_model(df, CONFIG)
print(clf_results["metrics"])
print(clf_results["report"])
```

## Cell 10 — Quick charts (optional, if you have matplotlib)
```python
import matplotlib.pyplot as plt

results["top_skills"].plot(kind="barh", x="skill", y="count", figsize=(8,6), legend=False)
plt.gca().invert_yaxis()
plt.title("Top 20 Requested Skills")
plt.tight_layout()
plt.show()
```
```python
results["salary_by_experience"].plot(kind="bar", x="experience_level", y="mean_salary", legend=False)
plt.title("Mean Salary by Experience Level")
plt.ylabel("Salary (USD)")
plt.tight_layout()
plt.show()
```

## Notes specific to your dataset
- `experience_level` values decoded: EN→Entry, MI→Mid, SE→Senior, EX→Executive
- `employment_type` decoded: FT→Full-time, PT→Part-time, CT→Contract, FL→Freelance
- `dataset_A` = ai_job_dataset.csv, `dataset_B` = ai_job_dataset1.csv (same 15,000 job_ids,
  but different attribute values per file — treated as two comparable samples, not duplicates)
- `salary_local` only exists in dataset_B; it's auto-dropped before modeling (kept in the
  raw dataframe if you want to inspect it — it's in `drop_for_model`, not deleted from `df`)
- On a normal laptop, `build_regression_model` / `build_classification_model` on the full
  30,000-row merged dataset should take well under a minute (150 trees, all CPU cores).
