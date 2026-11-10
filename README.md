livvy@fedora:~/Documents/Projects_to_be_finished/Global-AI-Job-Market-Intelligence-Report$ git add .gitignore models/
git commit -m "Ignore tooling folders; add trained model and metadata"
git push origin main
[main 4d0b193] Ignore tooling folders; add trained model and metadata
 3 files changed, 51 insertions(+), 1 deletion(-)
 create mode 100644 models/model_metadata.json
 create mode 100644 models/salary_prediction_pipeline.joblib
Enumerating objects: 8, done.
Counting objects: 100% (8/8), done.
Delta compression using up to 4 threads
Compressing objects: 100% (6/6), done.
Writing objects: 100% (6/6), 48.35 KiB | 12.09 MiB/s, done.
Total 6 (delta 2), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To https://github.com/Atieng/Global-AI-Job-Market-Intelligence-Report.git
   69b1c30..4d0b193  main -> main
livvy@fedora:~/Documents/Projects_to_be_finished/Global-AI-Job-Market-Intelligence-Report$ grep -iE "scikit|pandas|numpy|joblib|streamlit" requirements.txt
joblib==1.5.3
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
streamlit==1.61.1
livvy@fedora:~/Documents/Projects_to_be_finished/Global-AI-Job-Market-Intelligence-Report$ cat README.md
# Global AI Job Market Intelligence Report

A data science project that predicts salaries for AI & Data Science roles based on job characteristics — built end-to-end using the CRISP-DM methodology, from business understanding through a deployed interactive dashboard.

## Project Goal

Give job seekers, HR teams, recruiters, and universities a data-backed answer to *"what should this role actually pay?"* — based on real, measurable job characteristics (experience level, required skills, remote status, company size, industry) rather than guesswork or outdated salary surveys.

**Success criterion:** R² ≥ 0.70 on held-out test data.

## Process (CRISP-DM)

1. **Business Understanding** — Defined the problem, stakeholders, and success metrics.
2. **Data Understanding** — Explored dataset structure, distributions, and quality issues.
3. **Data Preparation** — Cleaned the data and engineered features (skill flags, seniority score, remote/hybrid indicators, etc.).
4. **Modeling** — Compared 5 regression algorithms, cross-validated the top performers, and tuned a Gradient Boosting model with `RandomizedSearchCV`.
5. **Evaluation** — Assessed model performance with MAE, RMSE, and R² on a held-out test set; identified and fixed a data leakage issue (a feature accidentally derived from the target variable) to ensure the final results are trustworthy.
6. **Deployment** — Saved the final model pipeline and built an interactive Streamlit dashboard for live predictions.

##  Repository Structure

```
├── Data/                 # Raw and/or processed dataset(s)
├── Streamlit/            # Streamlit dashboard app (salary_dashboard.py + saved model)
│   └── models/           # Saved model pipeline (.joblib) and metadata
├── python_files/         # [describe what this folder contains]
├── index.ipynb           # Main notebook — full CRISP-DM workflow
├── requirements.txt      # Python dependencies (pinned versions)
├── .gitignore
└── README.md
```

## Tech Stack

- **Python** — pandas, numpy
- **Modeling** — scikit-learn (Gradient Boosting, Random Forest, Decision Tree, Linear Regression, Ridge)
- **Visualization** — matplotlib, seaborn
- **Deployment** — Streamlit
- **Model persistence** — joblib

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Atieng/Global-AI-Job-Market-Intelligence-Report.git
cd Global-AI-Job-Market-Intelligence-Report
```

### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
Open `index.ipynb` in Jupyter to walk through the full analysis, from EDA to model evaluation.

### 5. Run the dashboard locally
```bash
cd Streamlit
streamlit run salary_dashboard.py
```

## Results

| Metric | Value |
|---|---|
| Model | Tuned Gradient Boosting Regressor |
| Test MAE | *[20374.819321]* |
| Test RMSE | *[29350.309721]* |
| Test R² | *[0.763807]* |

 

## 🌐 Live Demo

Try the deployed dashboard here: *[https://global-ai-job-market-intelligence-report-xnpdrkglaqym7wafldedl.streamlit.app/]*

##  Limitations

- The dataset reflects a snapshot in time and may not capture longer-term market shifts.
- Predictions are estimates, not guaranteed salary figures — use them as a benchmark range.

##  Notes

During evaluation, a data leakage issue was identified: a feature (`salary_category`) had been derived directly from the target variable (`salary_usd`) via quartile binning and accidentally left in the model's inputs. This was removed and the model retrained to ensure reported performance reflects genuine predictive power rather than an artifact of leakage.
 livvy@fedora:~/Documents/Projects_to_be_finished/Global-AI-Job-Market-Intelligence-Report$ 

