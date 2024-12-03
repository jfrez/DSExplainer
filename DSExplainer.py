import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
import itertools
from warnings import simplefilter
from scipy.sparse import csr_matrix

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class DSExplainer:
    def __init__(self, model, comb):
        self.model = model
        self.comb = comb
        self.explainer = shap.TreeExplainer(model)

    def generate_combinations(self, X):
        new_dataset = X.copy()
        for r in range(2, self.comb + 1):
            combinations = list(itertools.combinations(X.columns, r))
            for cols in combinations:
                new_col_name = "_x_".join(cols)
                new_dataset[new_col_name] = X[list(cols)].prod(axis=1)
        return new_dataset

    def ds_values(self, X):
        X = self.generate_combinations(X)
        shap_values = self.explainer.shap_values(X, check_additivity=False)

        shap_values_df = pd.DataFrame(
            shap_values,
            columns=X.columns,
            index=X.index
        )

        feature_columns = shap_values_df.columns

        normalized_shap = shap_values_df[feature_columns].abs()
        normalized_shap = normalized_shap.div(normalized_shap.sum(axis=1), axis=0)

        results = []

        for idx, row in normalized_shap.iterrows():
            masses = row.to_dict()
            certainty = {}
            plausibility = {}

            for k in masses.keys():
                hip = k.split("_x_")
                cert = 0
                for h in hip:
                    cert += masses[h]
                cert += masses[k]
                certainty[k] = cert

            for k in masses.keys():
                hip = k.split("_x_")
                plaus = 0

                for h_key, mass_value in masses.items():
                    related_hypotheses = h_key.split("_x_")
                    if any(h in hip for h in related_hypotheses):
                        plaus += mass_value

                plausibility[k] = plaus

            results.append({
                'Index': idx,
                'Certainty': certainty,
                'Plausibility': plausibility
            })

        certainty_df = pd.DataFrame([
            {**{'Index': res['Index']}, **res['Certainty']} for res in results
        ]).set_index('Index')
        plausibility_df = pd.DataFrame([
            {**{'Index': res['Index']}, **res['Plausibility']} for res in results
        ]).set_index('Index')

        return shap_values_df, certainty_df, plausibility_df
