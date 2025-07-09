import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from DSExplainer import DSExplainer
import numpy as np
import ollama
import os
from textwrap import dedent
import re
import time

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

# Language used to translate the LLM output. Can be overridden with the
# TRANSLATION_LANGUAGE environment variable.
TRANSLATION_LANGUAGE = os.getenv("TRANSLATION_LANGUAGE", "español")

# Load Iris dataset as pandas DataFrame
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
target_names = iris.target_names
original_features = X.copy()

# Scale all numerical features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train a tree-based model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the DSExplainer using the full dataset
max_comb = 3
explainer = DSExplainer(model, comb=max_comb, X=X_scaled, Y=y)
model = explainer.getModel()

# Generate DSExplainer outputs for a random sample from the full dataset
np.random.seed(int(time.time()) % 2**32)
subset = X_scaled.sample(n=1, random_state=np.random.randint(0, 10000))
orig_subset = original_features.loc[subset.index]

mass_values_df, certainty_df, plausibility_df = explainer.ds_values(subset)

# Generate predictions for the selected rows using the fitted scaler
X_pred = explainer.generate_combinations(subset, scaler=explainer.scaler)
raw_preds = model.predict(X_pred)
pred_labels = [target_names[int(round(p))] for p in raw_preds]

for df in (mass_values_df, certainty_df, plausibility_df):
    df["prediction"] = pred_labels

# Compare predictions with the original target values
true_labels = [target_names[t] for t in y.loc[subset.index]]


comparison_df = orig_subset.copy()
comparison_df["actual"] = true_labels
comparison_df["predicted"] = pred_labels

print("Original features with actual vs. predicted labels:")
print(comparison_df)

# Helper to print top values per row
TOP_N = 3

def print_top_columns(df, df_name):
    for idx, row in df.iterrows():
        numeric_row = row.drop(labels=["prediction"], errors="ignore")
        numeric_row = pd.to_numeric(numeric_row, errors="coerce")
        top_values = numeric_row.nlargest(TOP_N)
        print(f"\n{df_name}, Row {idx}:")
        for col, val in top_values.items():
            print(f"    {col}: {val}")

print_top_columns(mass_values_df, "mass_values_df")
print_top_columns(certainty_df, "certainty_df")
print_top_columns(plausibility_df, "plausibility_df")

# ----- LLM Interpretation -----
DATASET_DESCRIPTION = dedent(
    """
    The Iris dataset contains measurements of iris flowers. Each row provides
    the sepal length and width as well as petal length and width. The target
    variable indicates the species of the flower (setosa, versicolor or
    virginica).
    """
)

OBJECTIVE_DESCRIPTION = (
    "briefly conclude which species the sample belongs to. Only provide the final conclusion based on Certainty and Plausibility."
)


def resumen_fila(row_idx: int, top_n: int = TOP_N) -> str:
    pred = mass_values_df.loc[row_idx, "prediction"]

    cert_series = pd.to_numeric(
        certainty_df.drop(columns="prediction").iloc[row_idx], errors="coerce"
    )
    top_cert = cert_series.nlargest(top_n)
    cert_vals = ", ".join(top_cert.index)

    plaus_series = pd.to_numeric(
        plausibility_df.drop(columns="prediction").iloc[row_idx], errors="coerce"
    )
    top_plaus = plaus_series.nlargest(top_n)
    plaus_vals = ", ".join(top_plaus.index)

    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        f"Certainty values: {cert_vals}",
        f"Plausibility values: {plaus_vals}",
    ]

    return "\n".join(resumen)

for idx in range(len(mass_values_df)):
    features_text = ", ".join(
        f"{col}: {orig_subset.iloc[idx][col]}" for col in orig_subset.columns
    )

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

        translation_prompt = (
            f"Translate the following text to {TRANSLATION_LANGUAGE}:\n{clean}"
        )
        translated = llm_client.chat(
            model="mannix/jan-nano",
            messages=[{"role": "user", "content": translation_prompt}],
        ).message.content.strip()

        translated_clean = re.sub(
            r"<think>.*?</think>", "", translated, flags=re.DOTALL
        ).strip()

        print(f"\nLLM interpretation for row {idx} ({TRANSLATION_LANGUAGE}):")
        print(translated_clean)

    except Exception as e:
        print(f"\nCould not obtain LLM interpretation for row {idx}: {e}")
