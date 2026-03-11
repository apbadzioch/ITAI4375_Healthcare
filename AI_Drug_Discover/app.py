"""
Streamlit web app for predicting aqueous solubility using
a trained Random Forest model.

- Aqueous Solubility is the maximum amount of solute (solid, liquid, or gas) that dissolves in a given
  volume of water at a specific temperature and pressure.

1. Loads the pre-trained model
2. Accepts user inputs (MolWt and LogP)
3. generates a solubility prediction
"""

import streamlit as st
import joblib
import numpy as np

# load trained model
# Use Streamlit caching to avoid reloading the model every time the user interacts with the app.
@st.cache_resource
def load_model():
    """
    Load trained Random Forest model from disk.
    Cached for performance efficiency.
    """
    return joblib.load("solubility_model.joblib")

# load model
model = load_model()

# Streamlit UI Layout
st.title("AI-Powered Solubility Predictor")
st.write(
    """
    This application predicts aqueous solubility (log mol/L)
    using molecular descriptors:
    - Molecular Weight (MolWt)
    - LogP (lipophilicity)
    """
)

# --- USER INPUTS ---
# slider for Molecular Weight
mol_wt = st.slider(
    "Molecular Weight (MolWt)",
    min_value=50.0,
    max_value=900.0,
    value=200.0,
    step=0.1
)

# slider for LogP
log_p = st.slider(
    "LogP (lipophilicity)",
    min_value=-5.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

# --- GENERATE PREDICTION ---
input_features = np.array([[mol_wt, log_p]], dtype=float)

prediction = model.predict(input_features)[0]

# --- DISPLAY OUTPUT ---
st.subheader("Predicted Solubility")
st.write(
    f"Estimated log solubility: **{prediction:.3f} log(mol/L)**"
)

# Optional interpretation guidance
if prediction > 0:
    st.success("High predicted solubility")
elif prediction > -2:
    st.info("Moderate predicted solubility")
else:
    st.warning("Low predicted solubility")


"""

"""