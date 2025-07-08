import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from DSExplainer import DSExplainer
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

X_train, X_test, orig_train, orig_test, y_train, y_test = train_test_split(
    X, original_features, y, test_size=0.1, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama

    

max_comb = 3
explainer = DSExplainer(model, comb=max_comb,X=X_train,Y=y_train)
model = explainer.getModel()
np.random.seed(int(time.time()) % 2**32)  # Cambia semilla en cada ejecución
subset = X_test.sample(n=1, random_state=np.random.randint(0, 10000))
orig_subset = orig_test.loc[subset.index]
mass_values_df, certainty_df, plausibility_df = explainer.ds_values(subset)

# Generate predictions for the selected rows
X_pred = explainer.generate_combinations(subset)
raw_preds = model.predict(X_pred)
pred_labels = ["survived" if p >= 0.5 else "did not survive" for p in raw_preds]

for df in (mass_values_df, certainty_df, plausibility_df):
    df["prediction"] = pred_labels

# Compare predictions with the original target values
true_labels = [
    "survived" if t == 1 else "did not survive" for t in y_test.loc[subset.index]
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
        print(f"\n{df_name}, Fila {idx}:")
        for col, val in top_values.items():
            print(f"    {col}: {val}")


print_top_columns(certainty_df, "certainty_df")
print_top_columns(plausibility_df, "plausibility_df")

# ----- LLM Interpretation -----
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
    pred = mass_values_df.loc[row_idx, "prediction"]
    cert_vals = ", ".join(
        f"{k}: {v:.3f}" for k, v in certainty_df.drop(columns="prediction").iloc[row_idx].items()
    )
    plaus_vals = ", ".join(
        f"{k}: {v:.3f}" for k, v in plausibility_df.drop(columns="prediction").iloc[row_idx].items()
    )

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
        response = llm_client.chat(model="mannix/jan-nano", messages=[{"role": "user", "content": prompt}])
        clean = re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.DOTALL).strip()
        print(f"\nLLM interpretation for row {idx}:")
        print(clean)
    except Exception as e:
        print(f"\nCould not obtain LLM interpretation for row {idx}: {e}")
