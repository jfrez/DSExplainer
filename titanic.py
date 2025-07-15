import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from DSExplainer import DSExplainer
from sklearn.metrics import mean_absolute_error
import time
import numpy as np
from sklearn.datasets import fetch_openml
import ollama
import os
from textwrap import dedent
import re

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


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

# Language used to translate the LLM output. Can be overridden with the
# TRANSLATION_LANGUAGE environment variable.
TRANSLATION_LANGUAGE = os.getenv("TRANSLATION_LANGUAGE", "Spanish")

    

max_comb = 3
explainer = DSExplainer(model, comb=max_comb, X=X, Y=y)
model = explainer.getModel()

# Calculate model error on the training data
train_features = explainer.generate_combinations(X, scaler=explainer.scaler)
y_numeric = pd.to_numeric(y)
train_preds = model.predict(train_features)
target_range = y_numeric.max() - y_numeric.min()
model_error = mean_absolute_error(y_numeric, train_preds) / target_range
print(f"Model error rate: {model_error:.4f}")
np.random.seed(int(time.time()) % 2**32)  # Change seed for every run
subset = X.sample(n=1, random_state=np.random.randint(0, 10000))
orig_subset = original_features.loc[subset.index]

DATASET_DESCRIPTION = dedent(
    """
    The Titanic dataset contains details about passengers on the ill-fated ship
    and whether or not they survived.
    """
)

OBJECTIVE_SHAP = (
    "briefly conclude why the passenger survived or not based on SHAP features."
)

OBJECTIVE_DEMPSTER = (
    "briefly conclude why the passenger survived or not. Only provide the final conclusion based on Certainty and Plausibility."
)

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
    top_n=3,
    error_rate=model_error,
)

pred_labels = [
    "survived" if p >= 0.5 else "did not survive" for p in shap_values_df["prediction"]
]
for df in (mass_values_df, certainty_df, plausibility_df):
    df["prediction"] = pred_labels

# Compare predictions with the original target values
true_labels = [
    "survived" if t == 1 else "did not survive" for t in y.loc[subset.index]
]

comparison_df = orig_subset.copy()
comparison_df["actual"] = true_labels
comparison_df["predicted"] = pred_labels

print("Original features with actual vs. predicted labels:")
print(comparison_df)
 


top_n = 3


def print_top_columns(df, df_name):
    for idx, row in df.iterrows():
        numeric_row = row.drop(labels=["prediction"], errors="ignore")
        numeric_row = pd.to_numeric(numeric_row, errors="coerce")

        top_values = numeric_row.nlargest(top_n)
        print(f"\n{df_name}, Row {idx}:")
        for col, val in top_values.items():
            print(f"    {col}: {val}")


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
