
import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title='Student Placement Prediction', layout='centered')
st.title('🎓 Student Placement Prediction App')
st.write("Enter student details to predict whether they will get placed.")

MODEL_FILE = "model_pipeline.pkl"
FEATURE_FILE = "feature_columns.json"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully.")
else:
    model = None
    st.error("❌ Model file not found. Place model_pipeline.pkl in the same folder.")

if os.path.exists(FEATURE_FILE):
    feature_cols = json.load(open(FEATURE_FILE))
else:
    feature_cols = None
    st.error("❌ feature_columns.json not found. Place it in the same folder.")

st.header("🔍 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    MBA_Percentage           = st.number_input("MBA Percentage", 0.0, 100.0, 70.0)
    SSC_Percentage           = st.number_input("SSC Percentage", 0.0, 100.0, 80.0)
    HSC_Percentage           = st.number_input("HSC Percentage", 0.0, 100.0, 75.0)
    Graduation_Percentage    = st.number_input("Graduation Percentage", 0.0, 100.0, 72.0)
    Entrance_Score           = st.number_input("Entrance Score", 0.0, 100.0, 65.0)

with col2:
    Work_Experience_Years    = st.number_input("Work Experience (Years)", 0, 50, 1)
    Specialization           = st.selectbox("Specialization", ["Finance", "Marketing", "HR"])
    Communication_Marks      = st.number_input("Communication Marks", 0.0, 100.0, 75.0)
    BOCA_Marks               = st.number_input("BOCA Marks", 0.0, 100.0, 70.0)
    ProjectWork_Marks        = st.number_input("ProjectWork Marks", 0.0, 100.0, 78.0)

col3, col4 = st.columns(2)
with col3:
    Internship_Performance   = st.number_input("Internship Performance (0-10)", 0.0, 10.0, 7.0)
with col4:
    Analytical_Score         = st.number_input("Analytical Score", 0.0, 100.0, 68.0)

Age = st.number_input("Age", 15, 80, 24)

input_df = pd.DataFrame([{
    "MBA_Percentage": MBA_Percentage,
    "SSC_Percentage": SSC_Percentage,
    "HSC_Percentage": HSC_Percentage,
    "Graduation_Percentage": Graduation_Percentage,
    "Entrance_Score": Entrance_Score,
    "Work_Experience_Years": Work_Experience_Years,
    "Specialization": Specialization,
    "Communication_Marks": Communication_Marks,
    "BOCA_Marks": BOCA_Marks,
    "ProjectWork_Marks": ProjectWork_Marks,
    "Internship_Performance": Internship_Performance,
    "Analytical_Score": Analytical_Score,
    "Age": Age
}])


st.subheader("📋 Input Preview")
st.dataframe(input_df)

if st.button("🚀 Predict Placement"):

    if model is None or feature_cols is None:
        st.error("❌ Model or feature list missing.")
    else:
        X_pred = input_df.copy()
        # Ensure columns are in the expected order
        X_pred = X_pred[feature_cols]
        pred = model.predict(X_pred)[0]
        prob = model.predict_proba(X_pred)[0][1]

        st.subheader("📢 Prediction Result")
        if pred == 1:
            st.success(f"🎉 Student is LIKELY to be PLACED (Probability: {prob:.2f})")
        else:
            st.error(f"⚠️ Student is NOT likely to be placed (Probability: {prob:.2f})")
