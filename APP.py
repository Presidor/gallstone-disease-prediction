import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Gallstone Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺🏥 Gallstone Prediction App")
st.write("Predict Gallstone Disease using machine learning based on medical records.")

# -------------------------------
# FILE PATH SETUP (PRODUCTION SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "gallstone_prediction_model.pkl")
data_path = os.path.join(BASE_DIR, "gallstone_selected.csv")

# -------------------------------
# LOAD MODEL + DATA (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

model = load_model()
df = load_data()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🏥🩺 Medical Records")

vitamin_d = st.sidebar.slider("Vitamin D", 3.5, 53.1, 22.0)
alt = st.sidebar.slider("Alanin Aminotransferaz (ALT)", 3.0, 372.0, 19.0)
ast = st.sidebar.slider("Aspartat Aminotransferaz (AST)", 8.0, 195.0, 18.0)
bone_mass = st.sidebar.slider("Bone Mass (BM)", 1.4, 4.0, 2.8)
crp = st.sidebar.slider("C-Reactive Protein (CRP)", 0.0, 43.4, 0.2)
tbfr = st.sidebar.slider("Total Body Fat Ratio (%)", 6.3, 50.9, 27.8)
ecf_tbw = st.sidebar.slider("Extracellular Fluid/Total Body Water (ECF/TBW)", 29.2, 52.0, 42.0)
icw = st.sidebar.slider("Intracellular Water (ICW)", 13.8, 57.1, 23.0)
hemoglobin = st.sidebar.slider("Hemoglobin (HGB)", 8.5, 18.8, 14.4)
hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", [0, 1], index=0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Map gender to dataset encoding (0 = Male, 1 = Female)
gender_val = 0 if gender == "Male" else 1

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Predict Gallstone Status"):
    try:
        features = pd.DataFrame([{
            "Vitamin D": vitamin_d,
            "Alanin Aminotransferaz (ALT)": alt,
            "Aspartat Aminotransferaz (AST)": ast,
            "Bone Mass (BM)": bone_mass,
            "C-Reactive Protein (CRP)": crp,
            "Total Body Fat Ratio (TBFR) (%)": tbfr,
            "Extracellular Fluid/Total Body Water (ECF/TBW)": ecf_tbw,
            "Intracellular Water (ICW)": icw,
            "Hemoglobin (HGB)": hemoglobin,
            "Hyperlipidemia": hyperlipidemia,
            "Gender": gender_val
        }])

        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Prediction: Gallstones Present")
        else:
            st.info("❌ Prediction: No Gallstones Detected")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Machine Learning Gallstone Status Prediction")
