"""
main.py
=======
Runs the full CRISP-DM pipeline end-to-end:

  1. Business Understanding -> defined in config.py comments / your notes
  2. Data Understanding      -> data_prep.profile_data, eda.py
  3. Data Preparation        -> data_prep.clean_data / standardize_remote_column
  4. Modeling                -> modeling.build_regression_model / build_classification_model
  5. Evaluation               -> metrics printed + saved
  6. Deployment (light)      -> trained models saved to /models as .joblib

USAGE
-----
1. Edit config.py to point raw_data_path at your dataset and match column names.
2. Run:  python main.py
3. Find CSV summaries in /outputs and trained models in /models.

Reuse for a NEW dataset: copy this whole folder, edit config.py only.
"""

import os
import pandas as pd

from config import CONFIG
from data_prep import run_data_prep_merged
from eda import run_eda
from modeling import build_regression_model, build_classification_model


def save_eda_outputs(results: dict, cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    for name, val in results.items():
        if isinstance(val, pd.DataFrame):
            path = os.path.join(cfg.output_dir, f"{name}.csv")
            val.to_csv(path, index=False)
            print(f"[save_eda_outputs] Saved {path}")
        else:
            print(f"[save_eda_outputs] {name} = {val}")


def main():
    cfg = CONFIG
    print("=== STEP 1-3: Data Understanding & Preparation ===")
    df = run_data_prep_merged(cfg)

    print("\n=== STEP 4: EDA ===")
    eda_results = run_eda(df, cfg)
    save_eda_outputs(eda_results, cfg)

    print("\n=== STEP 5: Modeling - Regression (salary prediction) ===")
    reg_results = build_regression_model(df, cfg)

    print("\n=== STEP 6: Modeling - Classification (job level prediction) ===")
    clf_results = build_classification_model(df, cfg)

    print("\n=== DONE ===")
    print(f"Regression R2: {reg_results['metrics']['R2']:.3f}")
    print(f"Classification accuracy: {clf_results['metrics']['accuracy']:.3f}")
    print(f"Models saved in: {cfg.model_dir}")
    print(f"EDA tables saved in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
