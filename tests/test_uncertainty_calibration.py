import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss
import shap

from DSExplainer import DSExplainer

class UncertaintyCalibrationTest(unittest.TestCase):
    def setUp(self):
        cancer = load_breast_cancer(as_frame=True)
        X = cancer.data
        y = cancer.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

    def test_quantile_shap_and_brier(self):
        # Baseline model with SHAP
        model = RandomForestRegressor(n_estimators=50, random_state=0)
        model.fit(self.X_train, self.y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)

        # Bootstrapped SHAP for uncertainty estimation
        boot_vals = []
        for i in range(3):
            idx = np.random.choice(len(self.X_train), len(self.X_train), replace=True)
            X_res = self.X_train.iloc[idx]
            y_res = self.y_train.iloc[idx]
            m = RandomForestRegressor(n_estimators=50, random_state=i)
            m.fit(X_res, y_res)
            boot_vals.append(shap.TreeExplainer(m).shap_values(self.X_test))
        boot_vals = np.stack(boot_vals, axis=0)
        lower = np.quantile(boot_vals, 0.05, axis=0)
        upper = np.quantile(boot_vals, 0.95, axis=0)

        self.assertEqual(shap_values.shape, lower.shape)
        self.assertEqual(shap_values.shape, upper.shape)

        # DSExplainer for comparison
        ds_model = RandomForestRegressor(n_estimators=50, random_state=0)
        ds_explainer = DSExplainer(ds_model, comb=2, X=self.X_train, Y=self.y_train)
        shap_df, mass_df, cert_df, plaus_df = ds_explainer.ds_values(self.X_test)
        self.assertEqual(len(shap_df), len(self.X_test))
        self.assertIn("uncertainty", mass_df.columns)

        # Calibration metric (Brier score)
        preds = model.predict(self.X_test)
        brier = brier_score_loss(self.y_test, preds)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 1.0)

if __name__ == '__main__':
    unittest.main()
