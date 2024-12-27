from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
from joblib import load
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import PoissonRegressor 
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='dsTemplate')

# Load the trained models
ridge_model = load("ridge_random_search_model.pkl")
catboost_model = load("catboost_grid_search_model.pkl")  # Assuming saved as pkl using joblib
# Load the model
with open("poisson_model.pkl", "rb") as f:
    poisson_model = pickle.load(f)
ann_model = load_model("ann_bo.keras")
scaler = load('scaler.pkl')

# Define feature columns to exclude derived columns (e.g., probabilities)
all_columns = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",  # Example feature names; replace with actual names
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss", "InadequatePlanning",
    "PoliticalFactors"
]

feature_columns = [col for col in all_columns ]

@app.route('/')
def index():
    # Load the dataset and get the first 5 rows
    dataset = pd.read_csv("flood.csv")
    dataset_head = dataset.head()
    pca_images = {
        "Ridge": "/static/pca_ridge.png",
        "CatBoost": "/static/pca_catboost.png",
        "Poisson": "/static/pca_poisson.png",
        "ANN": "/static/pca_ann.png"
    }

    return render_template(
        'index.html',
        feature_columns=feature_columns,
        pca_images= pca_images,
        dataset_head=dataset_head
    )

# Function to clip predictions to [0, 1]
def clip_prediction(value):
    return max(0, min(value, 1))
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from the form
        features = []
        for feature_name in feature_columns:
            feature_value = request.form.get(feature_name)
            if feature_value:
                features.append(float(feature_value))
            else:
                return render_template('index.html', error=f"Missing value for {feature_name}", feature_columns=feature_columns)

        # Convert features to a Pandas DataFrame
        features_df = pd.DataFrame([features], columns=feature_columns)
        # Scale the input features
        features_df_scaled = scaler.transform(features_df)

        # Ridge Model Prediction
        ridge_pred = clip_prediction(ridge_model.predict(features_df)[0])
        
        # CatBoost Model Prediction
        catboost_pred = clip_prediction(catboost_model.predict(features_df)[0])
        
        # Poisson Model Prediction
        poisson_pred = clip_prediction(poisson_model.predict(features_df)[0])
        
        # ANN Model Prediction (Already probability-based, but clip for safety)
        logit_output = ann_model.predict(features_df_scaled)[0][0]
        ann_pred = clip_prediction(1 / (1 + np.exp(-logit_output)))
        pca_images = {
    "Ridge Model": "/static/pca_ridge.png",
    "CatBoost Model": "/static/pca_catboost.png",
    "Poisson Model": "/static/pca_poisson.png",
    "ANN Model": "/static/pca_ann.png"
}


        # Pass predictions to the template
        return render_template(
            'index.html',
            feature_columns=feature_columns,
            ridge_pred=round(ridge_pred, 3),
            catboost_pred=round(catboost_pred, 3),
            poisson_pred=round(poisson_pred, 3),
            ann_pred=round(ann_pred, 3),
            pca_images=pca_images
        )

    except ValueError:
        return render_template('index.html', error="Invalid input. Please enter numerical values.", feature_columns=feature_columns)
    except Exception as e:
        return render_template('index.html', error=str(e), feature_columns=feature_columns)

@app.route('/visualize_pca')
def visualize_pca():
    try:
        # Load dataset for visualization
        X_test = pd.read_csv("flood.csv")  # Replace with your actual test dataset
        X_test = X_test[feature_columns]  # Select feature columns

        # Initialize PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)

        # Generate predictions for all models
        ridge_predictions = ridge_model.predict(X_test)
        catboost_predictions = catboost_model.predict(X_test)
        poisson_predictions = poisson_model.predict(X_test)
        ann_predictions = ann_model.predict(X_test)
        ann_predictions = 1 / (1 + np.exp(-ann_predictions))  # Apply sigmoid transformation for ANN

        # Define a helper function to create and save PCA plots
        def create_pca_plot(predictions, title, filename):
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=predictions,
                cmap='viridis',
                alpha=0.7,
                edgecolors='k'
            )
            plt.colorbar(scatter, label="Predictions")
            plt.title(title)
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plt.savefig(f"static/{filename}")
            plt.close()

        # Create PCA visualizations for each model
        create_pca_plot(ridge_predictions, "PCA for Ridge Model Predictions", "pca_ridge.png")
        create_pca_plot(catboost_predictions, "PCA for CatBoost Model Predictions", "pca_catboost.png")
        create_pca_plot(poisson_predictions, "PCA for Poisson Model Predictions", "pca_poisson.png")
        create_pca_plot(ann_predictions, "PCA for ANN Model Predictions", "pca_ann.png")

        # Render the PCA results in a new template
        return render_template(
            'pca.html',
            pca_images={
                "Ridge": "/static/pca_ridge.png",
                "CatBoost": "/static/pca_catboost.png",
                "Poisson": "/static/pca_poisson.png",
                "ANN": "/static/pca_ann.png"
            }
        )

    except Exception as e:
        return f"Error in PCA Visualization: {str(e)}"
@app.route('/interactive_chart')
def interactive_chart():
    # Example: Comparing predictions for a subset of the dataset
    dataset = pd.read_csv("flood.csv")
    sample_data = dataset.sample(10)  # Take a random sample of 10 rows for visualization
    features = sample_data[feature_columns]
    
    # Generate predictions
    ridge_preds = ridge_model.predict(features)
    catboost_preds = catboost_model.predict(features)
    poisson_preds = poisson_model.predict(features)
    ann_preds = ann_model.predict(features)
    ann_preds = 1 / (1 + np.exp(-ann_preds))  # Apply sigmoid to ANN predictions

    # Create Plotly traces
    # Create Plotly traces
    trace_ridge = go.Scatter(
    x=list(range(1, len(sample_data) + 1)),  # Sequential sample labels
    y=ridge_preds,
    mode='lines+markers', name='Ridge Predictions',
    marker=dict(color='green')
    )
    trace_catboost = go.Scatter(
    x=list(range(1, len(sample_data) + 1)),
    y=catboost_preds,
    mode='lines+markers', name='CatBoost Predictions',
    marker=dict(color='blue')
    )
    trace_poisson = go.Scatter(
    x=list(range(1, len(sample_data) + 1)),
    y=poisson_preds,
    mode='lines+markers', name='Poisson Predictions',
    marker=dict(color='orange')
    )
    trace_ann = go.Scatter(
    x=list(range(1, len(sample_data) + 1)),
    y=ann_preds,
    mode='lines+markers', name='ANN Predictions',
    marker=dict(color='purple')
    )

    # Update the layout
    layout = go.Layout(
    title="Interactive Chart: Model Predictions",
    xaxis=dict(title="Sample Number"),  # Updated x-axis label
    yaxis=dict(title="Prediction Value"),
    hovermode="closest"
    )

    # Create the figure
    fig = go.Figure(data=[trace_ridge, trace_catboost, trace_poisson, trace_ann], layout=layout)
    chart_html = fig.to_html(full_html=False)

    # Render the chart in a template
    return render_template('chart.html', chart_html=chart_html)
if __name__ == '__main__':
    app.run(debug=True, port =5000)
