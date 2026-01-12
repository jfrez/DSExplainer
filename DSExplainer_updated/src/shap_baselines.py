import warnings
import numpy as np
import pandas as pd
from sklearn.utils import resample



def shap_bootstrap_intervals(explainer_obj, X, n_boot=200, alpha=0.05,
                          variant="absolute", random_state=0):
    rng = np.random.RandomState(random_state)


    shap_vals = explainer_obj.explainer.shap_values(X, check_additivity=False)


    if isinstance(shap_vals, list):
        if len(shap_vals) == 1:
            shap_vals_arr = shap_vals[0]
        else:
            try:
                shap_vals_arr = np.hstack(shap_vals)
            except ValueError:
                warnings.warn(
                    "SHAP values returned multiple arrays with non-hstackable shapes. Using only the first array."
                )
                shap_vals_arr = shap_vals[0]
    else:
        shap_vals_arr = shap_vals


    num_shap_features = shap_vals_arr.shape[1]
    if num_shap_features > len(X.columns):
        final_columns = list(X.columns) + [f"feature_{i}" for i in range(len(X.columns), num_shap_features)]
    elif num_shap_features < len(X.columns):
        final_columns = list(X.columns)[:num_shap_features]
    else:
        final_columns = list(X.columns)


    shap_df = pd.DataFrame(shap_vals_arr, columns=final_columns, index=X.index)


    def transform(row_vals):
        if variant == "absolute":
            return np.abs(row_vals)
        elif variant == "squared":
            return row_vals ** 2
        elif variant == "signed":
            return row_vals
        elif variant == "normalized":
            return row_vals / (np.sum(np.abs(row_vals)) + 1e-8)
        elif variant == "entropy":
            return -np.abs(row_vals) * np.log(np.abs(row_vals) + 1e-8)
        else:
            raise ValueError(f"Unknown variant: {variant}")


    low_rows, high_rows, mean_rows, width_rows = [], [], [], []
    for _, row in shap_df.iterrows():
        row_vals = row.values.astype(float)
        tvals = transform(row_vals)


        boots = []
        for _ in range(n_boot):
            boot = resample(tvals, replace=True, random_state=rng.randint(0, 10**9))
            boots.append(boot)


        boots = np.vstack(boots)
        lo = np.percentile(boots, 100 * alpha / 2, axis=0)
        hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)


        low_rows.append(lo)
        high_rows.append(hi)
        mean_rows.append(np.mean(boots, axis=0))
        width_rows.append(hi - lo)


    low = pd.DataFrame(low_rows, columns=shap_df.columns, index=X.index)
    high = pd.DataFrame(high_rows, columns=shap_df.columns, index=X.index)
    mean = pd.DataFrame(mean_rows, columns=shap_df.columns, index=X.index)
    width = pd.DataFrame(width_rows, columns=shap_df.columns, index=X.index)
    return mean, low, high, width