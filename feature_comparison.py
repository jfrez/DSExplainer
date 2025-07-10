import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from DSExplainer import DSExplainer
from textwrap import dedent
import ollama
import os
import re

# Optional LLM client
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama


def main(n_samples: int = 5):
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    data = titanic.frame.drop(columns=["boat", "name", "body", "home.dest"]).dropna()

    target = data["survived"]
    features = data.drop(columns=["survived"])
    original = features.copy()

    num_cols = features.select_dtypes(include=["number"]).columns
    cat_cols = features.columns.difference(num_cols)

    scaler = MinMaxScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])
    for col in cat_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col]).astype(int)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    explainer = DSExplainer(model, comb=3, X=features, Y=target)

    subset = features.sample(n=n_samples, random_state=42)
    original_subset = original.loc[subset.index]

    (
        shap_prompts,
        demp_prompts,
        shap_df,
        mass_df,
        cert_df,
        plaus_df,
    ) = explainer.ds_prompts(
        subset,
        original_subset,
        "The Titanic dataset contains passenger attributes and survival outcome.",
        "Summarize survival using SHAP features only.",
        "Summarize survival using Certainty and Plausibility metrics.",
        top_n=3,
        error_rate=0.0,
    )

    for idx in subset.index:
        print(f"\nRow {idx} top SHAP features vs DS hypotheses:")
        shap_lines = [ln for ln in shap_prompts[idx].splitlines() if ln.startswith("Top SHAP")]  # -> 'Top SHAP features: a, b, c'
        ds_lines = [ln for ln in demp_prompts[idx].splitlines() if ln.startswith("Certainty")]
        if shap_lines:
            print(shap_lines[0])
        if ds_lines:
            print(ds_lines[0])

        # Query LLM with the SHAP prompt
        try:
            resp = llm_client.chat(model="mannix/jan-nano", messages=[{"role": "user", "content": shap_prompts[idx]}])
            clean = re.sub(r"<think>.*?</think>", "", resp.message.content, flags=re.DOTALL).strip()
            print("LLM Response:")
            print(clean)
        except Exception as e:
            print(f"LLM error: {e}")


if __name__ == "__main__":
    main()
