import os
"""
AI/Data Science Salary Predictor — Streamlit Dashboard

Run with:
    streamlit run salary_dashboard.py

Expects the saved model pipeline at: models/salary_prediction_pipeline.joblib
(created in Section 6 of your notebook). Place this file in the same folder
as your notebook so the relative path resolves correctly.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💰", layout="centered")
st.title("💰 AI & Data Science Salary Predictor")
st.write(
    "Estimate a salary (USD) for an AI/Data Science role based on job characteristics. "
    "Powered by a tuned Gradient Boosting model."
)

# ---------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_prediction_pipeline.joblib")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run Section 6 of your notebook first to create "
        "`models/salary_prediction_pipeline.joblib`, then place that `models/` folder "
        "next to this script."
    )
    st.stop()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.header("Job Details")

col1, col2 = st.columns(2)

with col1:
    experience_level = st.selectbox(
        "Experience Level",
        options=["EN", "MI", "SE", "EX"],
        format_func=lambda x: {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}[x],
    )
    employment_type = st.selectbox(
        "Employment Type",
        options=["FT", "PT", "CT", "FL"],
        format_func=lambda x: {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}[x],
    )
    company_size = st.selectbox(
        "Company Size",
        options=["S", "M", "L"],
        format_func=lambda x: {"S": "Small", "M": "Medium", "L": "Large"}[x],
    )
    education_required = st.selectbox(
        "Education Required", options=["Associate", "Bachelor", "Master", "PhD"]
    )
    employee_residence = st.text_input("Employee Residence (country)", value="United States")
    industry = st.selectbox(
        "Industry",
        options=[
            "Technology", "Finance", "Healthcare", "Retail", "Education", "Manufacturing",
            "Consulting", "Media", "Automotive", "Real Estate", "Government",
            "Telecommunications", "Transportation", "Energy", "Gaming",
        ],
    )
    salary_currency = st.selectbox("Salary Currency", options=["USD", "EUR", "GBP"])

with col2:
    years_experience = st.slider("Years of Experience", 0, 20, 5)
    remote_ratio = st.selectbox(
        "Remote Ratio", options=[0, 50, 100],
        format_func=lambda x: {0: "On-site", 50: "Hybrid", 100: "Remote"}[x],
    )
    benefits_score = st.slider("Benefits Score", 0.0, 10.0, 5.0, step=0.1)
    job_description_length = st.slider("Job Description Length (chars)", 100, 3000, 1000)
    hiring_window_days = st.slider("Hiring Window (days)", 1, 120, 30)

st.subheader("Required Skills")
skill_cols = st.columns(4)
skill_names = ["Python", "PyTorch", "TensorFlow", "SQL", "AWS", "NLP", "Docker"]
skill_flags = {}
for i, skill in enumerate(skill_names):
    with skill_cols[i % 4]:
        skill_flags[f"has_{skill.lower()}"] = int(st.checkbox(skill, value=(skill == "Python")))

# ---------------------------------------------------------
# Derived features (must mirror the notebook's feature engineering)
# ---------------------------------------------------------
exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
size_map = {"S": 1, "M": 2, "L": 3}

experience_level_num = exp_map[experience_level]
company_size_num = size_map[company_size]
seniority_impact = experience_level_num * years_experience
skill_count = sum(skill_flags.values())
total_ai_skills = skill_count
is_remote = int(remote_ratio == 100)
is_hybrid = int(0 < remote_ratio < 100)

# NOTE: application_duration, posting_month/year, and deadline_month came from
# posting_date/application_deadline in the notebook. Since a live prediction has
# no real posting dates, we use reasonable defaults — adjust if your final
# feature set handles this differently.
application_duration = hiring_window_days
posting_month = 6
posting_year = 2025
deadline_month = "June"
experience_group = pd.cut(
    [years_experience], bins=[0, 2, 5, 10, 100],
    labels=["Entry", "Junior", "Mid", "Senior"], include_lowest=True
)[0]

# ---------------------------------------------------------
# Build the single-row input DataFrame
# Column set must exactly match X.columns used to train the pipeline
# (after the salary_category leakage fix — see notebook Section 3).
# ---------------------------------------------------------
input_data = pd.DataFrame([{
    "salary_currency": salary_currency,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "company_size": company_size,
    "employee_residence": employee_residence,
    "remote_ratio": remote_ratio,
    "education_required": education_required,
    "years_experience": years_experience,
    "industry": industry,
    "job_description_length": job_description_length,
    "benefits_score": benefits_score,
    "experience_level_num": experience_level_num,
    "company_size_num": company_size_num,
    **skill_flags,
    "seniority_impact": seniority_impact,
    "application_duration": application_duration,
    "posting_month": posting_month,
    "posting_year": posting_year,
    "deadline_month": deadline_month,
    "skill_count": skill_count,
    "total_ai_skills": total_ai_skills,
    "is_remote": is_remote,
    "is_hybrid": is_hybrid,
    "experience_group": experience_group,
    "hiring_window_days": hiring_window_days,
}])

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
st.divider()
if st.button("Predict Salary", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        mae = 13500  # approximate test-set MAE from the notebook — update if this changes after retraining
        low, high = prediction - mae, prediction + mae

        st.success(f"### Estimated Salary: ${prediction:,.0f} USD / year")
        st.write(f"**Typical range:** ${low:,.0f} – ${high:,.0f} / year")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the input columns don't exactly match what the model "
            "was trained on. Check your notebook's final `X.columns` and update the "
            "`input_data` dictionary above to match exactly."
        )