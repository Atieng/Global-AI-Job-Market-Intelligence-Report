# AI Job Market Analysis Pipeline (CRISP-DM, reusable)

A modular Python pipeline for analyzing AI/Data Science job market datasets
(skills demand, salary benchmarks, remote work trends, salary prediction,
job-level classification). Built so you can reuse it on future similar
datasets by editing **one file: `config.py`**.

## Structure

```
ai_job_market_pipeline/
├── config.py       # <-- EDIT THIS for each new dataset (column names, paths)
├── data_prep.py    # Data Understanding + Preparation (load, profile, clean)
├── eda.py          # Exploratory analysis: top skills, salary breakdowns, correlations
├── modeling.py     # Regression (salary) + Classification (job level) models
├── main.py         # Runs the full pipeline end-to-end
├── requirements.txt
├── data/           # put your raw CSV/Excel here
├── outputs/        # EDA result tables get saved here as CSV
└── models/         # trained models get saved here as .joblib
```

## How to use on a NEW dataset

1. Drop your dataset file into `data/`.
2. Open `config.py` and update:
   - `raw_data_path` → path to your file
   - `salary_col`, `country_col`, `experience_col`, `remote_col`, `skills_col`, `job_title_col`
     → rename to match your dataset's actual column headers
   - `regression_target` / `classification_target` if you want to predict something else
3. Run:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
4. Check `outputs/` for CSV summaries and `models/` for trained models.

You do **not** need to touch `data_prep.py`, `eda.py`, `modeling.py`, or
`main.py` for a new dataset — only `config.py`.

## What each stage covers (CRISP-DM)

| Stage | File | What it does |
|---|---|---|
| Business Understanding | (your notes / config comments) | Define target questions |
| Data Understanding | `data_prep.profile_data`, `eda.py` | Missingness, dtypes, skill frequency, salary stats, correlations |
| Data Preparation | `data_prep.clean_data`, `standardize_remote_column` | Dedup, type coercion, missing-value handling, category normalization |
| Modeling | `modeling.build_regression_model`, `build_classification_model` | RandomForest regression for salary, RandomForest classification for job level |
| Evaluation | printed metrics (MAE/RMSE/R², accuracy/F1) | Console + returned dict |
| Deployment (light) | `models/*.joblib` | Reload anytime with `modeling.load_model()` and call `predict_new()` |

## Using a trained model on new records later

```python
from modeling import load_model, predict_new
import pandas as pd

model = load_model("models/regression_model.joblib")
new_data = pd.DataFrame([...])  # same feature columns used in training
preds = predict_new(model, new_data)
```

## Notes

- `experience_salary_correlation()` assumes a rough seniority order
  (Entry < Mid < Senior < Executive). Adjust `order_guess` in `eda.py`
  if your dataset uses different labels.
- Skills column is assumed to be a single string per row with a separator
  (default `,`). If your dataset already has skills as a list/JSON, adjust
  `top_skills()` in `eda.py`.
- All modeling uses `RandomForest` for a strong, low-tuning baseline. Swap
  in other scikit-learn estimators inside `modeling.py` if needed.
