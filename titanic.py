import pandas as pd
from DSExplainer import DSExplainer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    from sklearn.datasets import fetch_openml
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    max_comb = 5
    explainer = DSExplainer(model, comb=max_comb)
    shap_values_df, certainty_df, plausibility_df = explainer.ds_values(X_test)

    print("SHAP Values DataFrame:")
    print(shap_values_df.head())
    print("Certainty DataFrame:")
    print(certainty_df.head())
    print("Plausibility DataFrame:")
    print(plausibility_df.head())

if __name__ == "__main__":
    main()
