# AI/Data Science Salary Predictor — Streamlit Dashboard
#
# Run with:
#     streamlit run salary_dashboard.py
#
# Expects the saved model pipeline at: models/salary_prediction_pipeline.joblib
# and metadata at: models/model_metadata.json (both created in Section 6 of
# your notebook). Place this file in the same folder so relative paths resolve.

import streamlit as st
import pandas as pd
import joblib
import json
import os

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Load model + metadata (cached so they only load once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/salary_prediction_pipeline.joblib")

@st.cache_data
def load_metadata():
    if os.path.exists("models/model_metadata.json"):
        with open("models/model_metadata.json") as f:
            return json.load(f)
    return None

try:
    model = load_model()
except FileNotFoundError:
    model = None

metadata = load_metadata()

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 💰 Salary Intelligence")
    st.caption("AI & Data Science Job Market")

    page = st.radio(
        "Navigate",
        options=["🏠 Overview", "🔮 Predict Salary", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data note")
    st.caption(
        "Predictions are estimates from a model trained on historical job "
        "postings — treat results as a benchmark range, not a guaranteed offer."
    )

# ---------------------------------------------------------
# Shared: model-not-found warning
# ---------------------------------------------------------
def require_model():
    if model is None:
        st.error(
            "Model file not found. Run Section 6 of your notebook first to create "
            "`models/salary_prediction_pipeline.joblib`, then place that `models/` "
            "folder next to this script."
        )
        st.stop()

# ===========================================================
# PAGE 1 — OVERVIEW
# ===========================================================
if page == "🏠 Overview":
    st.title("💰 AI & Data Science Salary Predictor")
    st.write(
        "Estimate salaries for AI/Data Science roles based on job characteristics, "
        "and see how the underlying model performs."
    )

    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    if metadata:
        col1.metric("Model", metadata.get("model_name", "—"))
        col2.metric("Test R²", f"{metadata.get('test_r2', 0):.3f}")
        col3.metric("Test MAE", f"${metadata.get('test_mae', 0):,.0f}")
        col4.metric("Test RMSE", f"${metadata.get('test_rmse', 0):,.0f}")
    else:
        col1.metric("Model", "Not loaded")
        col2.metric("Test R²", "—")
        col3.metric("Test MAE", "—")
        col4.metric("Test RMSE", "—")
        st.warning(
            "Model metadata not found. Run the Section 6.2 cell in your notebook "
            "to generate `models/model_metadata.json` for these stats to appear."
        )

    st.divider()

    st.subheader("How to use this app")
    st.markdown(
        "- Go to **🔮 Predict Salary** to enter job details and get a live salary estimate.\n"
        "- Go to **📈 Model Performance** to see how accurate and reliable the model is.\n"
        "- Go to **ℹ️ About** for details on the data, methodology, and known limitations."
    )

    # ---------------------------------------------------------
    # Trend charts from raw data (optional — only shows if CSV is present)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Market Trends")

    DATA_PATH = "../Data/ai_job_dataset.csv"  # confirmed via `find . -iname "*.csv"`
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        t1, t2 = st.columns(2)
        with t1:
            st.write("**Salary distribution**")
            salary_bins = pd.cut(df["salary_usd"], bins=10).value_counts().sort_index()
            salary_bins.index = salary_bins.index.astype(str)  # Altair can't render Interval objects
            st.bar_chart(salary_bins)
        with t2:
            st.write("**Average salary by experience level**")
            order = ["EN", "MI", "SE", "EX"]
            avg_by_exp = df.groupby("experience_level")["salary_usd"].mean().reindex(order)
            st.bar_chart(avg_by_exp)
    else:
        st.info(
            f"Place your dataset CSV at `{DATA_PATH}` (adjust the path in the script "
            "to match your actual file) to unlock salary distribution and trend charts here."
        )

# ===========================================================
# PAGE 2 — PREDICT SALARY
# ===========================================================
elif page == "🔮 Predict Salary":
    require_model()

    st.title("🔮 Predict a Salary")
    st.write("Fill in the job details below to get an estimated annual salary (USD).")

    st.divider()
    st.subheader("Job Details")

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
    skill_cols = st.columns(7)
    skill_names = ["Python", "PyTorch", "TensorFlow", "SQL", "AWS", "NLP", "Docker"]
    skill_flags = {}
    for i, skill in enumerate(skill_names):
        with skill_cols[i]:
            skill_flags[f"has_{skill.lower()}"] = int(st.checkbox(skill, value=(skill == "Python")))

    # Derived features (must mirror the notebook's feature engineering)
    exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
    size_map = {"S": 1, "M": 2, "L": 3}

    experience_level_num = exp_map[experience_level]
    company_size_num = size_map[company_size]
    seniority_impact = experience_level_num * years_experience
    skill_count = sum(skill_flags.values())
    total_ai_skills = skill_count
    is_remote = int(remote_ratio == 100)
    is_hybrid = int(0 < remote_ratio < 100)

    posting_month = 6
    posting_year = 2025
    deadline_month = "June"
    experience_group = pd.cut(
        [years_experience], bins=[0, 2, 5, 10, 100],
        labels=["Entry", "Junior", "Mid", "Senior"], include_lowest=True
    )[0]

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

    # Initialize prediction history in session state
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.divider()
    if st.button("Predict Salary", type="primary", use_container_width=True):
        try:
            prediction = model.predict(input_data)[0]
            mae = metadata.get("test_mae", 13500) if metadata else 13500
            low, high = prediction - mae, prediction + mae

            r1, r2 = st.columns(2)
            r1.metric("Estimated Salary", f"${prediction:,.0f} / year")
            r2.metric("Typical Range", f"${low:,.0f} – ${high:,.0f}")

            # Save to session history (most recent first, capped at 10)
            st.session_state.prediction_history.insert(0, {
                "Experience": {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}[experience_level],
                "Years Exp.": years_experience,
                "Industry": industry,
                "Remote": {0: "On-site", 50: "Hybrid", 100: "Remote"}[remote_ratio],
                "Skills": skill_count,
                "Predicted Salary": f"${prediction:,.0f}",
            })
            st.session_state.prediction_history = st.session_state.prediction_history[:10]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "This usually means the input columns don't exactly match what the model "
                "was trained on. Check your notebook's final `X.columns` and update the "
                "`input_data` dictionary above to match exactly."
            )

    # Show prediction history for this session
    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Prediction History (this session)")
        st.dataframe(
            pd.DataFrame(st.session_state.prediction_history),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state.prediction_history = []
            st.rerun()

# ===========================================================
# PAGE 3 — MODEL PERFORMANCE
# ===========================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if metadata:
        st.subheader("Test Set Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{metadata.get('test_r2', 0):.3f}")
        col2.metric("MAE", f"${metadata.get('test_mae', 0):,.0f}")
        col3.metric("RMSE", f"${metadata.get('test_rmse', 0):,.0f}")

        st.divider()
        st.subheader("Best Hyperparameters")
        best_params = metadata.get("best_params", {})
        if best_params:
            st.json(best_params)
        else:
            st.caption("No hyperparameter details found in metadata.")

        st.divider()
        st.subheader("Features Used")
        features = metadata.get("features", [])
        if features:
            st.write(f"The model uses **{len(features)}** input features:")
            st.code(", ".join(features), language=None)

        st.divider()
        st.subheader("Feature Importance")
        try:
            preprocessor = model.named_steps["preprocessor"]
            gb_model = model.named_steps["model"]

            cat_cols = preprocessor.transformers_[1][2]
            num_cols = preprocessor.transformers_[0][2]

            ohe = preprocessor.named_transformers_["cat"]
            ohe_feature_names = ohe.get_feature_names_out(cat_cols)
            all_feature_names = list(num_cols) + list(ohe_feature_names)

            importances = gb_model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": all_feature_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=False).head(15)

            st.bar_chart(feat_df.set_index("Feature"))
            st.caption("Top 15 features driving salary predictions, ranked by the model's internal importance scores.")

            st.markdown(
                "**What this means:** Experience-related signals — years of experience, "
                "experience level, and their interaction — typically dominate salary "
                "predictions in this model, confirming that seniority is the single "
                "biggest driver of compensation. Location (certain high-cost-of-living "
                "countries) and company size tend to matter next, at a noticeably smaller "
                "scale. Specific technical skills usually rank lower than seniority factors, "
                "suggesting *how experienced you are* outweighs *which exact tools you know*."
            )
        except Exception as e:
            st.info(
                f"Feature importance couldn't be computed automatically ({e}). "
                "This can happen if the pipeline's step names differ from `preprocessor`/`model`, "
                "or if the model isn't loaded."
            )
    else:
        st.warning(
            "No model metadata found. Run the Section 6.2 cell in your notebook to "
            "generate `models/model_metadata.json`."
        )

# ===========================================================
# PAGE 4 — ABOUT
# ===========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        This dashboard estimates salaries for AI & Data Science job postings using a
        machine learning model trained on historical job market data.

        **Process (CRISP-DM):**
        1. Business Understanding — defined the problem and stakeholders
        2. Data Understanding — explored dataset structure and quality
        3. Data Preparation — cleaned data and engineered features
        4. Modeling — compared regression models, tuned a Gradient Boosting Regressor
        5. Evaluation — validated on a held-out test set, corrected a data leakage issue
        6. Deployment — this dashboard

        **Limitations:**
        - Predictions are estimates, not guaranteed salary figures.
        - The underlying data reflects a snapshot in time and may not capture
          longer-term market shifts.
        """
    )