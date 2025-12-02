
import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np

st.set_page_config(page_title='Student Placement Prediction (portable)', layout='centered')
st.title('🎓 Student Placement Prediction (portable model)')
st.write("This app uses a lightweight numpy-based logistic model to avoid scikit-learn version issues.")

MODEL_FILE = "model_simple.pkl"
FEATURE_FILE = "feature_columns.json"

model = None
model_is_dict = False

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        if isinstance(model, dict) and model.get("model_type") == "numpy_logistic":
            model_is_dict = True
            st.success("✅ Portable model loaded (numpy logistic).")
        else:
            st.warning("⚠️ Loaded model is not the portable format. If it's a scikit-learn object, it may fail to load on some hosts.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ model_simple.pkl not found. Place it in the same folder.")

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

# Build input dataframe in original form
input_base = {
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
}

st.subheader("📋 Input Preview")
st.json(input_base)

if st.button("🚀 Predict Placement"):
    if model is None or feature_cols is None:
        st.error("❌ Model or feature list missing.")
    elif model_is_dict:
        # Convert input into feature vector matching feature_order
        f = model["feature_order"]
        # create one-hot specialization
        spec = input_base["Specialization"]
        spec_vec = {
            "Finance": [1.0, 0.0, 0.0],
            "Marketing": [0.0, 1.0, 0.0],
            "HR": [0.0, 0.0, 1.0]
        }[spec]
        arr = np.array([
            input_base["MBA_Percentage"],
            input_base["SSC_Percentage"],
            input_base["HSC_Percentage"],
            input_base["Graduation_Percentage"],
            input_base["Entrance_Score"],
            input_base["Work_Experience_Years"],
            spec_vec[0],
            spec_vec[1],
            spec_vec[2],
            input_base["Communication_Marks"],
            input_base["BOCA_Marks"],
            input_base["ProjectWork_Marks"],
            input_base["Internship_Performance"],
            input_base["Analytical_Score"],
            input_base["Age"]
        ], dtype=float)

        # standardize
        x_mean = np.array(model["x_mean"])
        x_std = np.array(model["x_std"])
        x_stdzd = (arr - x_mean) / x_std

        w = np.array(model["weights"])
        b = float(model["intercept"])
        logit = x_stdzd.dot(w) + b
        prob = 1 / (1 + np.exp(-logit))
        pred = int(prob >= 0.5)

        st.subheader("📢 Prediction Result")
        if pred == 1:
            st.success(f"🎉 Student is LIKELY to be PLACED (Probability: {prob:.2f})")
        else:
            st.error(f"⚠️ Student is NOT likely to be placed (Probability: {prob:.2f})")
    else:
        # Try to use sklearn model if it's there
        try:
            import pandas as pd
            X_pred = pd.DataFrame([input_base])
            # Basic preprocessing for specialization to match older pipeline if present
            X_pred = X_pred.assign(
                Specialization= X_pred["Specialization"]
            )
            # If it's a sklearn pipeline, attempt to predict
            pred = model.predict(X_pred)[0]
            prob = model.predict_proba(X_pred)[0][1]
            if pred == 1:
                st.success(f"🎉 Student is LIKELY to be PLACED (Probability: {prob:.2f})")
            else:
                st.error(f"⚠️ Student is NOT likely to be placed (Probability: {prob:.2f})")
        except Exception as e:
            st.error(f"❌ Could not use loaded model: {e}")
