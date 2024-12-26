import streamlit as st
from joblib import load
import numpy as np
from tensorflow.keras.models import load_model

# Load all models
models = {
    "ANN (Bayesian Optimization)": load_model("ann_bo.h5"),
    "Poisson (Randomized Search)": load("best_poisson_model.pkl"),
    "Ridge (Randomized Search)": load("ridge_random_search_model.pkl"),
    "CatBoost (Grid Search)": load("catboost_grid_search_model.pkl")
}

# Title and description
st.title("Flood Prediction App")
st.write("""
This application predicts flood probabilities using multiple machine learning models. 
It displays predictions from ANN, Poisson, Ridge, and CatBoost methods simultaneously.
""")

# Input fields for features (customize based on your dataset)
st.write("### Input Features")
monsoon_intensity = st.number_input("Monsoon Intensity (1-10):", min_value=1, max_value=10, step=1)
topography_drainage = st.number_input("Topography Drainage (1-10):", min_value=1, max_value=10, step=1)
urbanization = st.number_input("Urbanization (1-10):", min_value=1, max_value=10, step=1)
climate_change = st.number_input("Climate Change Impact (1-10):", min_value=1, max_value=10, step=1)

# Collect features into a numpy array
features = np.array([[monsoon_intensity, topography_drainage, urbanization, climate_change]])

# Prediction button
if st.button("Predict Flood Probability"):
    st.write("### Prediction Results")
    for model_name, model in models.items():
        if model_name == "ANN (Bayesian Optimization)":
            # ANN expects input data to be preprocessed similarly to training
            prediction = model.predict(features)
            st.write(f"**{model_name}:** Predicted Flood Probability = {prediction[0][0]:.3f}")
        else:
            # Other models use .pkl format
            prediction = model.predict(features)
            st.write(f"**{model_name}:** Predicted Flood Probability = {prediction[0]:.3f}")

# Add footer
st.write("---")
st.write("**Streamlit URL:** Add this URL after deployment.")
