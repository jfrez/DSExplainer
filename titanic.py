import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from DSExplainer import DSExplainer

from sklearn.datasets import fetch_openml
import ollama
from textwrap import dedent
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
    

max_comb = 3
explainer = DSExplainer(model, comb=max_comb,X=X_train,Y=y_train)
model = explainer.getModel()
mass_values_df, certainty_df, plausibility_df = explainer.ds_values(X_test[:2])
 


top_n = 3  


def print_top_columns(df, df_name):
    for idx, row in df.iterrows():
        top_values = row.nlargest(top_n)
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

FEATURES_TEXT = ", ".join(X.columns)


def resumen_fila(row_idx: int) -> str:
    X_row = explainer.generate_combinations(X_test.iloc[[row_idx]])
    pred = model.predict(X_row)[0]
    mass_top = mass_values_df.iloc[row_idx].nlargest(top_n)
    cert_top = certainty_df.iloc[row_idx].nlargest(top_n)
    plaus_top = plausibility_df.iloc[row_idx].nlargest(top_n)

    resumen = [
        f"Prediction for row {row_idx}: {pred}",
        "Top combinations by mass: "
        + ", ".join(f"{k} ({v:.3f})" for k, v in mass_top.items()),
        "Highest certainty in: "
        + ", ".join(f"{k} ({v:.3f})" for k, v in cert_top.items()),
        "Greatest plausibility in: "
        + ", ".join(f"{k} ({v:.3f})" for k, v in plaus_top.items()),
    ]

    return "\n".join(resumen)


for idx in range(len(mass_values_df)):
    prompt = (
        DATASET_DESCRIPTION
        + f"\nColumns: {FEATURES_TEXT}\n"
        + resumen_fila(idx)
        + "\nProvide an English interpretation of this inference result."
    )

    try:
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        print(f"\nLLM interpretation for row {idx}:")
        print(response.message.content)
    except Exception as e:
        print(f"\nCould not obtain LLM interpretation for row {idx}: {e}")
