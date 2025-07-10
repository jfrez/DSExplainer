import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss
import shap

from DSExplainer import DSExplainer


def main(seed: int = 42, bootstraps: int = 5):
    # Load data and split
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # Baseline model
    base_model = RandomForestRegressor(n_estimators=100, random_state=seed)
    base_model.fit(X_train_scaled, y_train)
    base_explainer = shap.TreeExplainer(base_model)
    shap_values = base_explainer.shap_values(X_test_scaled)

    # Bootstrapped SHAP for uncertainty
    boot_vals = []
    for i in range(bootstraps):
        idx = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_res = X_train_scaled.iloc[idx]
        y_res = y_train.iloc[idx]
        m = RandomForestRegressor(n_estimators=100, random_state=i)
        m.fit(X_res, y_res)
        expl = shap.TreeExplainer(m)
        boot_vals.append(expl.shap_values(X_test_scaled))
    boot_vals = np.stack(boot_vals, axis=0)
    lower = np.quantile(boot_vals, 0.05, axis=0)
    upper = np.quantile(boot_vals, 0.95, axis=0)

    # DSExplainer variant
    ds_model = RandomForestRegressor(n_estimators=100, random_state=seed)
    ds_explainer = DSExplainer(ds_model, comb=2, X=X_train_scaled, Y=y_train)
    shap_df, mass_df, cert_df, plaus_df = ds_explainer.ds_values(X_test_scaled)

    # Brier scores
    base_preds = base_model.predict(X_test_scaled)
    base_brier = brier_score_loss(y_test, base_preds)

    ds_preds = ds_model.predict(
        ds_explainer.generate_combinations(X_test_scaled, scaler=ds_explainer.scaler)
    )
    ds_brier = brier_score_loss(y_test, ds_preds)

    print("Ablation Results (Brier score lower is better):")
    print(f"1. Baseline SHAP: {base_brier:.4f}")
    print(
        "2. Bootstrapped SHAP (5th-95th quantile for first sample):",
        lower[0, :3],
        upper[0, :3],
    )
    print(f"3. DSExplainer: {ds_brier:.4f}")


if __name__ == "__main__":
    main()
