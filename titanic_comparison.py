import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from DSExplainer import DSExplainer
import numpy as np
from sklearn.datasets import fetch_openml
import shap
import ollama
import os
from textwrap import dedent
import re
import time
import sys

# Redirect stdout to a log file line by line
sys.stdout = open("log.txt", "w", buffering=1)




titanic = fetch_openml('titanic', version=1, as_frame=True)
data = titanic.frame
data = data.drop(columns=['boat','name', 'body', 'home.dest'])
data = data.dropna()

target_column = 'survived'
target = data[target_column]
features = data.drop(columns=[target_column])
original_features = features.copy()

numerical_columns = features.select_dtypes(include=['number']).columns
categorical_columns = features.columns.difference(numerical_columns)

scaler = MinMaxScaler()
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
for col in categorical_columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col]).astype(int)

X = features
y = target

model = RandomForestRegressor(n_estimators=100, random_state=42)

max_comb = 3
explainer = DSExplainer(model, comb=max_comb, X=X, Y=y)
model = explainer.getModel()

np.random.seed(int(time.time()) % 2**32)
subset = X.sample(n=20, random_state=np.random.randint(0, 100000))
orig_subset = original_features.loc[subset.index]

shap_values_df, certainty_df, plausibility_df = explainer.ds_values(subset)

X_pred = explainer.generate_combinations(subset)
raw_preds = model.predict(X_pred)
pred_labels = ["survived" if p >= 0.5 else "did not survive" for p in raw_preds]
for df in (shap_values_df, certainty_df, plausibility_df):
    df["prediction"] = pred_labels

TOP_N = 3

def get_top_features(df):
    """Return the top ``TOP_N`` columns with their original values."""
    top_dict = {}
    for idx, row in df.iterrows():
        numeric_row = row.drop(labels=["prediction"], errors="ignore")
        numeric_row = pd.to_numeric(numeric_row, errors="coerce")
        top_series = numeric_row.abs().nlargest(TOP_N)
        top_dict[idx] = [
            (col, row[col]) for col in top_series.index
        ]
    return top_dict

shap_top = get_top_features(shap_values_df[original_features.columns])
# Only consider combination columns for Dempster--Shafer metrics
combo_cols = [c for c in certainty_df.columns if "_x_" in c]
certainty_top = get_top_features(certainty_df[combo_cols])
plausibility_top = get_top_features(plausibility_df[combo_cols])

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

DATASET_DESCRIPTION = dedent(
    """
    The Titanic dataset contains information about passengers on the famous ship.
    Each row represents a passenger and includes variables such as ticket class
    (`pclass`), sex, age, number of siblings/spouses (`sibsp`) and parents/
    children (`parch`) aboard, the fare paid, cabin, and embarkation port. The
    target variable is `survived`, indicating whether the passenger lived.
    """
)

OBJECTIVE_SHAP = (
    "briefly conclude why the passenger survived or not. "
    "Only provide the final conclusion based on SHAP feature importances."
)

OBJECTIVE_DEMPSTER = (
    "briefly conclude why the passenger survived or not. "
    "Only provide the final conclusion based on Certainty and Plausibility."
)


def resumen_shap(row_idx: int) -> str:
    pred = shap_values_df.loc[row_idx, "prediction"]
    shap_vals = ", ".join(
        f"{name}: {val:.3f}" for name, val in shap_top[row_idx]
    )
    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        f"Top SHAP features: {shap_vals}",
    ]
    return "\n".join(resumen)


def resumen_dempster(row_idx: int) -> str:
    pred = shap_values_df.loc[row_idx, "prediction"]
    cert_vals = ", ".join(
        f"{name}: {val:.3f}" for name, val in certainty_top[row_idx]
    )
    plaus_vals = ", ".join(
        f"{name}: {val:.3f}" for name, val in plausibility_top[row_idx]
    )
    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        f"Certainty triples: {cert_vals}",
        f"Plausibility triples: {plaus_vals}",
    ]
    return "\n".join(resumen)


for idx in range(len(shap_values_df)):
    features_text = ", ".join(
        f"{col}: {orig_subset.iloc[idx][col]}" for col in orig_subset.columns
    )

    # ---- SHAP interpretation ----
    shap_prompt = (
        DATASET_DESCRIPTION
        + f"\nObjective: {OBJECTIVE_SHAP}"
        + f"\nColumns: {features_text}\n"
        + resumen_shap(idx)
    )
    print(shap_prompt)
    try:
        shap_response = llm_client.chat(
            model="mannix/jan-nano",
            messages=[{"role": "user", "content": shap_prompt}],
        )
        clean_shap = re.sub(
            r"<think>.*?</think>", "", shap_response.message.content, flags=re.DOTALL
        ).strip()
        print(f"\nLLM SHAP interpretation for row {idx} (English):")
        print(clean_shap)
    except Exception as e:
        print(f"\nCould not obtain SHAP interpretation for row {idx}: {e}")

    # ---- Dempster-Shafer interpretation ----
    demp_prompt = (
        DATASET_DESCRIPTION
        + f"\nObjective: {OBJECTIVE_DEMPSTER}"
        + f"\nColumns: {features_text}\n"
        + resumen_dempster(idx)
    )
    print(demp_prompt)
    try:
        demp_response = llm_client.chat(
            model="mannix/jan-nano",
            messages=[{"role": "user", "content": demp_prompt}],
        )
        clean_demp = re.sub(
            r"<think>.*?</think>", "", demp_response.message.content, flags=re.DOTALL
        ).strip()
        print(f"\nLLM Dempster interpretation for row {idx} (English):")
        print(clean_demp)
    except Exception as e:
        print(f"\nCould not obtain Dempster interpretation for row {idx}: {e}")

