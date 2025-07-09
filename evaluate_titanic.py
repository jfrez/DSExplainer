import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from DSExplainer import DSExplainer
from sklearn.datasets import fetch_openml
import numpy as np
import ollama
import os
from textwrap import dedent
import re

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

titanic = fetch_openml('titanic', version=1, as_frame=True)
data = titanic.frame
data = data.drop(columns=['boat', 'name', 'body', 'home.dest'])
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
explainer = DSExplainer(model, comb=3, X=X, Y=y)
model = explainer.getModel()

train_features = explainer.generate_combinations(X, scaler=explainer.scaler)
y_numeric = pd.to_numeric(y)
train_preds = model.predict(train_features)
model_error = mean_absolute_error(y_numeric, train_preds) / (y_numeric.max() - y_numeric.min())

subset = X.sample(n=10, random_state=42)
orig_subset = original_features.loc[subset.index]

DATASET_DESCRIPTION = dedent(
    """
    The Titanic dataset contains details about passengers on the ill-fated ship and whether they survived.
    """
)
OBJECTIVE_SHAP = "briefly conclude why the passenger survived or not based on SHAP features."
OBJECTIVE_DEMPSTER = (
    "Explain whether the passenger survived using the Certainty and "
    "Plausibility metrics. Focus on these metrics and avoid stating the "
    "exact survival percentage."
)

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

# Remove raw prediction values from the prompts to avoid
# conveying survival percentages directly.
for idx, text in demp_prompts.items():
    lines = [ln for ln in text.splitlines() if not ln.startswith("Prediction for row")]
    demp_prompts[idx] = "\n".join(lines)

pred_labels = ["survived" if p >= 0.5 else "did not survive" for p in shap_df["prediction"]]
true_labels = ["survived" if t == 1 else "did not survive" for t in y.loc[subset.index]]
comparison_df = orig_subset.copy()
comparison_df["actual"] = true_labels
comparison_df["predicted"] = pred_labels
print(comparison_df)

for idx, prompt in demp_prompts.items():
    print(f"Prompt {idx}:\n{prompt}\n")
    try:
        resp = llm_client.chat(model="mannix/jan-nano", messages=[{"role": "user", "content": prompt}])
        clean_resp = re.sub(r"<think>.*?</think>", "", resp.message.content, flags=re.DOTALL).strip()
        print("LLM:", clean_resp)
    except Exception as e:
        print("LLM error:", e)
