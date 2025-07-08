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
    top_dict = {}
    for idx, row in df.iterrows():
        numeric_row = row.drop(labels=["prediction"], errors="ignore")
        numeric_row = pd.to_numeric(numeric_row, errors="coerce")
        top_series = numeric_row.abs().nlargest(TOP_N)
        top_dict[idx] = list(top_series.index)
    return top_dict

shap_top = get_top_features(shap_values_df)
certainty_top = get_top_features(certainty_df)
plausibility_top = get_top_features(plausibility_df)

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

OBJECTIVE_DESCRIPTION = (
    "briefly conclude why the passenger survived or not. Only provide the final conclusion based on Certainty and Plausibility."
)


def resumen_fila(row_idx: int) -> str:
    pred = shap_values_df.loc[row_idx, "prediction"]
    shap_vals = ", ".join(shap_top[row_idx])
    cert_vals = ", ".join(certainty_top[row_idx])
    plaus_vals = ", ".join(plausibility_top[row_idx])
    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        f"Top SHAP values: {shap_vals}",
        f"Certainty values: {cert_vals}",
        f"Plausibility values: {plaus_vals}",
    ]
    return "\n".join(resumen)

for idx in range(len(shap_values_df)):
    features_text = ", ".join(f"{col}: {orig_subset.iloc[idx][col]}" for col in orig_subset.columns)
    prompt = (
        DATASET_DESCRIPTION
        + f"\nObjective: {OBJECTIVE_DESCRIPTION}"
        + f"\nColumns: {features_text}\n"
        + resumen_fila(idx)
    )
    print(prompt)
    try:
        response = llm_client.chat(
            model="mannix/jan-nano", messages=[{"role": "user", "content": prompt}]
        )
        clean = re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.DOTALL).strip()
        print(f"\nLLM interpretation for row {idx} (English):")
        print(clean)
    except Exception as e:
        print(f"\nCould not obtain LLM interpretation for row {idx}: {e}")

