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

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
target_names = iris.target_names
original_features = X.copy()

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

model = RandomForestRegressor(n_estimators=100, random_state=42)
explainer = DSExplainer(model, comb=3, X=X_scaled, Y=y)
model = explainer.getModel()

train_features = explainer.generate_combinations(X_scaled, scaler=explainer.scaler)
train_preds = model.predict(train_features)
model_error = mean_absolute_error(y, train_preds) / (y.max() - y.min())

subset = X_scaled.sample(n=10, random_state=42)
orig_subset = original_features.loc[subset.index]

DATASET_DESCRIPTION = dedent(
    """
    The Iris dataset contains measurements of iris flowers and the species to which each sample belongs.
    """
)
OBJECTIVE_SHAP = "briefly conclude which species the sample belongs to based on SHAP features."
OBJECTIVE_DEMPSTER = (
    "Explain which species the sample belongs to using the Certainty and "
    "Plausibility metrics as key evidence."
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

pred_labels = [target_names[int(round(p))] for p in shap_df["prediction"]]
true_labels = [target_names[t] for t in y.loc[subset.index]]
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
