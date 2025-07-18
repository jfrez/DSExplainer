import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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
TRANSLATION_LANGUAGE = os.getenv("TRANSLATION_LANGUAGE", "Spanish")

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

# Calculate model error on the training data
train_features = explainer.generate_combinations(X_scaled, scaler=explainer.scaler)
train_preds = model.predict(train_features)
target_range = y.max() - y.min()
model_error = mean_absolute_error(y, train_preds) / target_range
print(f"Model error rate: {model_error:.4f}")

# Generate DSExplainer outputs for a random sample from the full dataset
np.random.seed(int(time.time()) % 2**32)
subset = X_scaled.sample(n=1, random_state=np.random.randint(0, 10000))
orig_subset = original_features.loc[subset.index]

DATASET_DESCRIPTION = dedent(
    """
    The Iris dataset contains measurements of iris flowers and the species to
    which each sample belongs.
    """
)


OBJECTIVE_SHAP = dedent("""
    Write a single descriptive paragraph that explains which species the sample belongs to, using the SHAP metrics as the main sources of evidence.
    Focus on the most significant feature combinations (top values). Refer to the input feature values (e.g., petal width, sepal length) where relevant.
    Do not include bullet points or headings. Be concise but informative, and use a technical tone.
    End with a clear classification of the sample (e.g., 'The sample is classified as versicolor.').
""")
OBJECTIVE_DEMPSTER = dedent("""
    Write a single descriptive paragraph that explains which species the sample belongs to, using the Certainty and Plausibility metrics as the main sources of evidence.
    Focus on the most significant feature combinations (top values). Refer to the input feature values (e.g., petal width, sepal length) where relevant.
    Do not include bullet points or headings. Be concise but informative, and use a technical tone.
    End with a clear classification of the sample (e.g., 'The sample is classified as versicolor.').
""")


TOP_N = 3

(
    shap_prompts,
    demp_prompts,
    shap_values_df,
    mass_values_df,
    certainty_df,
    plausibility_df,
) = explainer.ds_prompts(
    subset,
    orig_subset,
    DATASET_DESCRIPTION,
    OBJECTIVE_SHAP,
    OBJECTIVE_DEMPSTER,
    top_n=TOP_N,
    error_rate=model_error,
)

pred_labels = [target_names[int(round(p))] for p in shap_values_df["prediction"]]
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


for idx, prompt in demp_prompts.items():
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
