import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
from itertools import combinations

from warnings import simplefilter
from scipy.sparse import csr_matrix

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class DSExplainer:
    def __init__(self, model, comb,X,Y):
        self.model = model
        self.comb = comb
        X = self.generate_combinations(X)
        model.fit(X, Y)
        self.explainer = shap.TreeExplainer(model)
        
    def getModel(self):
        return self.model

    def generate_combinations(self, X):
        new_dataset = X.copy()
        
        # Generate combinations of columns and add their sums to the dataset
        for r in range(2, self.comb + 1):
           new_columns = [
              (pd.Series(X[list(cols)].sum(axis=1), name="_x_".join(cols)))
                  for cols in combinations(X.columns, r)]

        new_dataset = pd.concat([new_dataset] + new_columns, axis=1)
                
        # Scale the dataset using MinMaxScaler
        scaler = MinMaxScaler()
        new_dataset = pd.DataFrame(scaler.fit_transform(new_dataset), columns=new_dataset.columns)
        
        return new_dataset

    def ds_values(self, X):
        X=self.generate_combinations(X)
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

                #certainity of the hypotessis k
                cert = 0
                for h in hip:
                    cert += masses[h]
                cert += masses[k]
                certainty[k] = cert

                #plausibility of the hypotessis k
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
