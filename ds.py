from flask import Flask, render_template, request
import plotly.graph_objs as go
from joblib import load
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import pickle

app = Flask(__name__, template_folder="dsTemplates", static_folder="static")

# Load models
ridge_model = load("ridge_random_search_model.pkl")
catboost_model = load("catboost_grid_search_model.pkl")
with open("poisson_model.pkl", "rb") as f:
    poisson_model = pickle.load(f)
ann_model = load_model("ann_bo.keras")
scaler = load("scaler.pkl")

# Feature columns
feature_columns = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning",
    "PoliticalFactors"
]

@app.route('/')
def index():
    dataset = pd.read_csv("flood.csv").head()

    # Generate PCA dynamically
    def generate_pca(model, name, X, y):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        predictions = model.predict(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions, cmap='viridis', edgecolor='k')
        plt.title(f"PCA Visualization - {name}")
        plt.colorbar(label='Predictions')
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        file_path = f"static/pca_{name.lower()}.png"
        plt.savefig(file_path)
        plt.close()
        return file_path

    X = pd.read_csv("flood.csv")[feature_columns]
    pca_images = {
        "Ridge": generate_pca(ridge_model, "Ridge", X, None),
        "CatBoost": generate_pca(catboost_model, "CatBoost", X, None),
        "Poisson": generate_pca(poisson_model, "Poisson", X, None),
        "ANN": generate_pca(ann_model, "ANN", scaler.transform(X), None),
    }

    return render_template(
        "index.html",
        feature_columns=feature_columns,
        dataset_head=dataset,
        pca_images=pca_images
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        features = [float(request.form.get(col, 0)) for col in feature_columns]
        features_df = pd.DataFrame([features], columns=feature_columns)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predictions
        ridge_pred = ridge_model.predict(features_df)[0]
        catboost_pred = catboost_model.predict(features_df)[0]
        poisson_pred = poisson_model.predict(features_df)[0]
        ann_pred = 1 / (1 + np.exp(-ann_model.predict(features_scaled)[0][0]))

        return render_template(
            "index.html",
            feature_columns=feature_columns,
            ridge_pred=round(ridge_pred, 3),
            catboost_pred=round(catboost_pred, 3),
            poisson_pred=round(poisson_pred, 3),
            ann_pred=round(ann_pred, 3)
        )

    except Exception as e:
        return render_template("index.html", error=str(e), feature_columns=feature_columns)

@app.route('/interactive_chart')
def interactive_chart():
    dataset = pd.read_csv("flood.csv").sample(10)
    X = dataset[feature_columns]

    ridge_preds = ridge_model.predict(X)
    catboost_preds = catboost_model.predict(X)
    poisson_preds = poisson_model.predict(X)
    ann_preds = 1 / (1 + np.exp(-ann_model.predict(scaler.transform(X))))

    trace_ridge = go.Scatter(x=list(range(len(ridge_preds))), y=ridge_preds, mode='lines+markers', name='Ridge')
    trace_catboost = go.Scatter(x=list(range(len(catboost_preds))), y=catboost_preds, mode='lines+markers', name='CatBoost')
    trace_poisson = go.Scatter(x=list(range(len(poisson_preds))), y=poisson_preds, mode='lines+markers', name='Poisson')
    trace_ann = go.Scatter(x=list(range(len(ann_preds))), y=ann_preds, mode='lines+markers', name='ANN')

    layout = go.Layout(title="Interactive Model Comparison", xaxis=dict(title="Sample"), yaxis=dict(title="Prediction"))
    fig = go.Figure(data=[trace_ridge, trace_catboost, trace_poisson, trace_ann], layout=layout)

    return render_template("chart.html", chart_html=fig.to_html(full_html=False))

if __name__ == "__main__":
    app.run(debug=True)
