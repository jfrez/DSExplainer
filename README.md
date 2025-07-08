# DSExplainer

## Overview

`DSExplainer` is a Python module designed to provide an extended interpretation of machine learning models by leveraging SHAP (SHapley Additive exPlanations) values. It adds an additional layer of understanding through **certainty** and **plausibility** metrics, computed based on combinations of features from the dataset.

The core idea behind `DSExplainer` is to combine features in a dataset and analyze their effects on model predictions, not just individually but also in combination. This approach helps highlight how interactions between features can impact model outcomes, providing more context compared to traditional feature importance metrics.

## Installation

To use `DSExplainer`, ensure you have Python installed along with the required dependencies. You can install the dependencies using the following command:

```bash
pip install pandas statsmodels scikit-learn shap
```

## Usage

### Importing the Module

First, make sure that the `DSExplainer.py` module is available in your project directory. Import it as follows:

```python
from DSExplainer import DSExplainer
```

### Example Usage

The example below demonstrates how to use `DSExplainer` with the Titanic dataset to train a `RandomForestRegressor` and analyze its predictions:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
# The fitted MinMaxScaler is stored in the explainer and reused for new data
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
```

### Parameters

- `model`: A un-trained machine learning model (the DSExplainer will do the fit). Currently, `DSExplainer` is designed to work with tree-based models compatible with SHAP.
- `comb`: An integer representing the maximum number of features to be combined. This parameter controls how many features are considered in feature interactions (e.g., pairs, triplets).

### Methods

- `ds_values(X)`: Generates SHAP values, certainty, and plausibility metrics for the input dataset `X`.
  - **Returns**: Three pandas DataFrames: `shap_values_df`, `certainty_df`, and `plausibility_df`.

## Theory Behind DSExplainer

The `DSExplainer` aims to extend the concept of feature importance using combinations of features. The two main metrics introduced are:

1. **Certainty**: This metric is computed to understand how certain a prediction is, considering the combined influence of related features. It sums the SHAP masses of each hypothesis that involves a given feature or combination of features.

2. **Plausibility**: Plausibility is calculated to understand the likelihood of different feature combinations influencing the output. It includes the mass of features that share related influences, providing insight into how plausible different feature combinations are in affecting model predictions.

These metrics can help uncover not just which features are important, but also how their combinations influence outcomes, offering deeper insights into the decision-making process of the model.

## Contributing

If you'd like to contribute to `DSExplainer`, feel free to submit issues or create pull requests. Contributions are always welcome!

## License

This project is licensed under the MIT License.

