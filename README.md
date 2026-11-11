# Global AI Job Market Intelligence Report

A data science project that predicts salaries for AI and Data Science roles from job characteristics. It follows the CRISP-DM methodology end to end, from business understanding through a deployed interactive dashboard.

**[Live demo](https://global-ai-job-market-intelligence-report-xnpdrkglaqym7wafldedl.streamlit.app/)**

## Project Goal

Give job seekers, HR teams, recruiters, and universities a data-backed answer to *"what should this role pay?"*, based on measurable job characteristics (experience level, required skills, remote status, company size, industry) rather than guesswork or outdated salary surveys.

**Success criterion:** R² ≥ 0.70 on held-out test data. **Achieved: R² = 0.764.**

## Results

| Metric | Value |
|---|---|
| Model | Tuned Gradient Boosting Regressor |
| Test MAE | $20,035 |
| Test RMSE | $29,350 |
| Test R² | 0.764 |
| 5-fold CV R² (training set) | 0.765 (std 0.010) |

Test and cross-validation scores agree closely, so the model is not overfitting.

### What these numbers mean

- **R² = 0.76:** the model explains about three-quarters of the differences in salary between jobs. The rest comes from factors not in the data, such as negotiation and individual circumstances.
- **Average error ≈ $20,000 (about 17-20% of a typical salary):** a predicted salary of $100,000 should be read as roughly $80,000-$120,000.
- **Versus guessing:** always guessing the average salary gives errors of around $60,000. This model roughly halves that.
- **Best used as:** a benchmark range for salary conversations, not an exact quote.

**Model comparison (test set, default settings):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | $20,310 | $29,501 | 0.761 |
| Random Forest | $20,649 | $30,413 | 0.746 |
| Ridge Regression | $21,113 | $30,445 | 0.746 |
| Extra Trees | $23,035 | $34,539 | 0.673 |
| Decision Tree | $27,602 | $41,395 | 0.530 |

**What drives salary:** experience level and years of experience dominate the feature importances, followed by the seniority interaction feature, employee country of residence, and company size.

## Process (CRISP-DM)

1. **Business Understanding:** defined the problem, stakeholders, and success metrics.
2. **Data Understanding:** explored dataset structure, distributions, and quality.
3. **Data Preparation:** cleaned the data and engineered features (skill flags, seniority score, remote/hybrid indicators, hiring window).
4. **Modeling:** compared 5 regression algorithms, cross-validated the top performers, and tuned Gradient Boosting and Random Forest with `RandomizedSearchCV`.
5. **Evaluation:** assessed MAE, RMSE, and R² on a held-out test set. Found and fixed a data leakage issue (see Notes) so the reported results are trustworthy.
6. **Deployment:** saved the final pipeline and built an interactive Streamlit dashboard for live predictions.

## Repository Structure

```
├── Data/                 # Raw dataset files
├── Streamlit/            # Streamlit dashboard (streamlit_app.py)
│   └── models/           # Model pipeline (.joblib) and metadata used by the app
├── models/               # Saved model pipeline (.joblib) and metadata from the notebook
├── python_files/         # Supporting Python scripts
├── index.ipynb           # Main notebook: full CRISP-DM workflow
├── requirements.txt      # Python dependencies (pinned versions)
├── .gitignore
└── README.md
```

## Tech Stack

- **Data:** pandas, numpy
- **Modeling:** scikit-learn (Gradient Boosting, Random Forest, Extra Trees, Decision Tree, Ridge)
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit
- **Model persistence:** joblib

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
streamlit run streamlit_app.py
```

## Data

[Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/) (Kaggle): 15,000 AI and Data Science job postings with salary, experience level, skills, location, company size, and remote ratio.

## Limitations

- **The dataset appears to be synthetic or heavily simplified.** Relationships in the data are unusually clean (for example, a 0.93 correlation between years of experience and experience level), so performance on real-world postings would likely be lower.
- **Performance plateaus around R² ≈ 0.76.** A linear Ridge baseline (R² 0.746) performs almost as well as the tuned ensemble (R² 0.764), which suggests the remaining variance is noise rather than something a more complex model can capture.
- The data is a single snapshot and does not capture longer-term market shifts.
- Predictions are estimates, not guaranteed salary figures. Use them as a benchmark range. The average error is about $20,000.

## Notes

During evaluation, a data leakage issue was identified: a feature (`salary_category`) had been derived directly from the target variable (`salary_usd`) via quartile binning and left in the model's inputs. It was removed and the model retrained. This lowered the headline score from an inflated R² ≈ 0.87 to a genuine R² ≈ 0.76, and the figures in this README reflect the corrected model.

## Author

Ivy Atieno. [GitHub](https://github.com/Atieng)
