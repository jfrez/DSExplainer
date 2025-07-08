import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from DSExplainer import DSExplainer

# Load Iris dataset as pandas DataFrame
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Scale all numerical features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# Train a tree-based model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the DSExplainer
max_comb = 3
explainer = DSExplainer(model, comb=max_comb, X=X_train, Y=y_train)
model = explainer.getModel()

# Generate DSExplainer outputs for the first two test samples
mass_values_df, certainty_df, plausibility_df = explainer.ds_values(X_test[:2])

# Helper to print top values per row
TOP_N = 3

def print_top_columns(df, df_name):
    for idx, row in df.iterrows():
        top_values = row.nlargest(TOP_N)
        print(f"\n{df_name}, Row {idx}:")
        for col, val in top_values.items():
            print(f"    {col}: {val}")

print_top_columns(mass_values_df, "mass_values_df")
print_top_columns(certainty_df, "certainty_df")
print_top_columns(plausibility_df, "plausibility_df")
