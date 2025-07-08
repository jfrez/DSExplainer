import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from DSExplainer import DSExplainer

from sklearn.datasets import fetch_openml
import ollama
import os
from textwrap import dedent
import re
titanic = fetch_openml('titanic', version=1, as_frame=True)
data = titanic.frame
data = data.drop(columns=['boat', 'body', 'home.dest'])
data = data.dropna()  

target_column = 'survived'
target = data[target_column]
features = data.drop(columns=[target_column])

numerical_columns = features.select_dtypes(include=['number']).columns
categorical_columns = features.columns.difference(numerical_columns)

scaler = MinMaxScaler()
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
for col in categorical_columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col]).astype(int)

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
llm_client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama
    

max_comb = 3
explainer = DSExplainer(model, comb=max_comb,X=X_train,Y=y_train)
model = explainer.getModel()
subset = X_test[:2]
mass_values_df, certainty_df, plausibility_df = explainer.ds_values(subset)

# Generate predictions for the same rows and append them to each result DataFrame
X_pred = explainer.generate_combinations(subset)
raw_preds = model.predict(X_pred)
pred_labels = ["survived" if p >= 0.5 else "did not survive" for p in raw_preds]
for df in (mass_values_df, certainty_df, plausibility_df):
    df["prediction"] = pred_labels
 


top_n = 3  


def print_top_columns(df, df_name):
    for idx, row in df.iterrows():
        numeric_row = row.drop(labels=["prediction"], errors="ignore")
        numeric_row = pd.to_numeric(numeric_row, errors="coerce")
        top_values = numeric_row.nlargest(top_n)
        print(f"\n{df_name}, Fila {idx}:")
        for col, val in top_values.items():
            print(f"    {col}: {val}")


print_top_columns(mass_values_df, "mass_values_df")
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

OBJECTIVE_DESCRIPTION = "Explain why the passenger survived or not based on the DSExplainer metrics."

FEATURES_TEXT = ", ".join(X.columns)


def resumen_fila(row_idx: int) -> str:
    pred = mass_values_df.loc[row_idx, "prediction"]

    mass_vals = ", ".join(
        f"{k}: {v:.3f}" for k, v in mass_values_df.drop(columns="prediction").iloc[row_idx].items()
    )
    cert_vals = ", ".join(
        f"{k}: {v:.3f}" for k, v in certainty_df.drop(columns="prediction").iloc[row_idx].items()
    )
    plaus_vals = ", ".join(
        f"{k}: {v:.3f}" for k, v in plausibility_df.drop(columns="prediction").iloc[row_idx].items()
    )

    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        f"Mass values: {mass_vals}",
        f"Certainty values: {cert_vals}",
        f"Plausibility values: {plaus_vals}",
    ]

    return "\n".join(resumen)


for idx in range(len(mass_values_df)):
    prompt = (
        DATASET_DESCRIPTION
        + f"\nObjective: {OBJECTIVE_DESCRIPTION}"
        + f"\nColumns: {FEATURES_TEXT}\n"
        + resumen_fila(idx)
    )

    try:
        response = llm_client.chat(model="mannix/jan-nano", messages=[{"role": "user", "content": prompt}])
        clean = re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.DOTALL).strip()
        print(f"\nLLM interpretation for row {idx}:")
        print(clean)
    except Exception as e:
        print(f"\nCould not obtain LLM interpretation for row {idx}: {e}")
