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
    """Wrapper around SHAP that adds Dempster–Shafer style metrics.

    Parameters
    ----------
    model : object
        A tree based estimator compatible with ``shap.TreeExplainer``.
    comb : int
        Maximum size of feature combinations to generate.
    X : pandas.DataFrame
        Training features used to fit ``model``.
    Y : pandas.Series or array-like
        Target values used to fit ``model``.
    """
    def __init__(self, model, comb, X, Y):


        """Fit the estimator and prepare the SHAP explainer.

        Parameters
        ----------
        model : object
            Estimator that will be fitted on the provided data.
        comb : int
            Maximum size of feature combinations to generate.
        X : pandas.DataFrame
            Feature matrix used to fit ``model``.
        Y : pandas.Series or array-like
            Target values corresponding to ``X``.
        """
        self.model = model
        self.comb = comb
        self.scaler = None

        X = self.generate_combinations(X)
        model.fit(X, Y)
        self.explainer = shap.TreeExplainer(model)
        
    def getModel(self):
        """Return the underlying fitted model."""
        return self.model

    def generate_combinations(self, X, scaler=None):

        """Create scaled columns for feature combinations.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset for which the combinations will be generated.

        Returns
        -------
        pandas.DataFrame
            A new dataset containing the original features, their
            combinations up to ``self.comb`` and scaled between 0 and 1.
        """

        new_dataset = X.copy()

        # Generate combinations of columns and add their sums to the dataset
        for r in range(2, self.comb + 1):
            for cols in combinations(X.columns, r):
                new_col_name = "_x_".join(cols)
                new_dataset[new_col_name] = X[list(cols)].sum(axis=1)

        # Scale the dataset using the provided scaler or fit a new one
        if scaler is None:
            scaler = MinMaxScaler()
            self.scaler = scaler
            scaler.fit(new_dataset)

        new_dataset = pd.DataFrame(scaler.transform(new_dataset), columns=new_dataset.columns)

        return new_dataset

    def ds_values(self, X):
        """Compute SHAP masses, certainty and plausibility for ``X``.

        The method first generates the same feature combinations that
        were used to fit the model. It then obtains SHAP values and
        normalizes them into "masses". These masses are used to derive
        Dempster–Shafer certainty and plausibility measures for each
        hypothesis (feature or feature combination).

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset for which the explanations are generated.

        Returns
        -------
        tuple of pandas.DataFrame
            ``shap_values_df`` with raw SHAP values, ``certainty_df`` and
            ``plausibility_df`` containing Dempster–Shafer metrics.
        """
        X = self.generate_combinations(X, scaler=self.scaler)

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
