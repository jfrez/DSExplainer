import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from DSExplainer import DSExplainer
import numpy as np
from textwrap import dedent
import ollama
import os
import re

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

# Load dataset
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target
original_features = X.copy()

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Fit model through DSExplainer
model = RandomForestRegressor(n_estimators=100, random_state=42)
explainer = DSExplainer(model, comb=3, X=X_scaled, Y=y)
model = explainer.getModel()

# Model error on training data for uncertainty mass
train_features = explainer.generate_combinations(X_scaled, scaler=explainer.scaler)
train_preds = model.predict(train_features)
model_error = mean_absolute_error(y, train_preds) / (y.max() - y.min())

# Sample subset for prompts
subset = X_scaled.sample(n=10, random_state=42)
orig_subset = original_features.loc[subset.index]

DATASET_DESCRIPTION = dedent(
    """
    The Breast Cancer Wisconsin (Diagnostic) dataset contains numerical measurements of cell nuclei obtained from digitized images of breast tissue biopsies. Each observation corresponds to a tumor and includes features such as radius, texture, perimeter, area, smoothness, and concavity, among others. Each sample is labeled as either malignant or benign.
    """
)

OBJECTIVE_SHAP = dedent("""
    Write a single descriptive paragraph that explains whether the tumor is malignant or benign, using SHAP feature importances as the main source of evidence.
    Emphasize the most relevant features contributing to the prediction (e.g., worst radius, mean concave points), and refer to input values where appropriate.
    Maintain a technical tone, avoid bullet points or headings, and keep the explanation concise.
    Conclude with a clear classification (e.g., 'The tumor is classified as malignant.').
""")

OBJECTIVE_DEMPSTER = dedent("""
    Write a single descriptive paragraph that explains whether the tumor is malignant or benign, based on the Certainty and Plausibility metrics.
    Focus on the most influential feature combinations with high certainty or plausibility scores, and refer to specific input values when relevant.
    Use a technical and concise tone. Do not include bullet points or section titles.
    Conclude with a clear statement on the tumor’s classification (e.g., 'The tumor is predicted to be benign.').
""")


(
    shap_prompts,
    demp_prompts,
    shap_df,
    mass_df,
    cert_df,
    plaus_df,
) = explainer.ds_prompts(
    subset,
    orig_subset,
    DATASET_DESCRIPTION,
    OBJECTIVE_SHAP,
    OBJECTIVE_DEMPSTER,
    top_n=3,
    error_rate=model_error,
)

pred_labels = ["malignant" if p >= 0.5 else "benign" for p in shap_df["prediction"]]
true_labels = ["malignant" if t == 1 else "benign" for t in y.loc[subset.index]]
comparison_df = orig_subset.copy()
comparison_df["actual"] = true_labels
comparison_df["predicted"] = pred_labels
print(comparison_df)

for idx, prompt in demp_prompts.items():
    print(f"Prompt {idx}:\n{prompt}\n")
    try:
        resp = llm_client.chat(model="mannix/jan-nano", messages=[{"role": "user", "content": prompt}])
        clean_resp = re.sub(r"<think>.*?</think>", "", resp.message.content, flags=re.DOTALL).strip()
        print("--- LLM RESPONSE ---")
        print(clean_resp)
        print("--------------------\n")
    except Exception as e:
        print("LLM error:", e)
