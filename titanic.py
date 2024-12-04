import pandas as pd
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
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    

    max_comb = 3
    explainer = DSExplainer(model, comb=max_comb,X=X_train,Y=y_train)
    model = explainer.getModel()
    mass_values_df, certainty_df, plausibility_df = explainer.ds_values(X_test)

if __name__ == "__main__":
    main()
