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

        Notes
        -----
        A :class:`~sklearn.preprocessing.MinMaxScaler` is created and fitted
        on ``X`` (including generated combinations) and stored as
        :pyattr:`self.scaler` for reuse on new data.
        """

        self.model = model
        self.comb = comb
        # Scaler stored for reuse on new data
        self.scaler = MinMaxScaler()

        # Generate feature combinations and fit the scaler
        X = self.generate_combinations(X, scaler=self.scaler)
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
        scaler : sklearn.preprocessing.MinMaxScaler, optional
            Scaler used to normalize the columns. If ``None`` the scaler
            stored in the explainer is used.

        Returns
        -------
        pandas.DataFrame
            A new dataset containing the original features, their
            combinations up to ``self.comb`` and scaled between 0 and 1.
        """


        new_dataset = X.copy()

        # Generate combinations of columns and add their sums to the dataset
        for r in range(2, self.comb + 1):
            new_columns = [
                pd.Series(X[list(cols)].sum(axis=1), name="_x_".join(cols))
                for cols in combinations(X.columns, r)
            ]
            new_dataset = pd.concat([new_dataset] + new_columns, axis=1)
                
        # Use the provided scaler or fall back to the stored one
        scaler = scaler or self.scaler
        if scaler is None:
            scaler = MinMaxScaler()
            self.scaler = scaler

        if hasattr(scaler, "n_samples_seen_"):
            new_dataset = pd.DataFrame(
                scaler.transform(new_dataset),
                columns=new_dataset.columns,
                index=new_dataset.index,
            )
        else:
            new_dataset = pd.DataFrame(
                scaler.fit_transform(new_dataset),
                columns=new_dataset.columns,
                index=new_dataset.index,
            )
        


        return new_dataset

    def ds_values(self, X, error_rate=0.0):
        """Compute SHAP masses, certainty and plausibility for ``X``.

        The method first generates the same feature combinations that
        were used to fit the model. It then obtains SHAP values and
        normalizes them into "masses". These masses are used to derive
        Dempster–Shafer certainty and plausibility measures for each
        hypothesis (feature or feature combination).

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset for which the explanations are generated. It will be
            transformed using the stored scaler.
        error_rate : float, optional
            Percentage of model error expressed as a fraction between
            0 and 1. This value is treated as the Dempster\u2013Shafer
            "uncertainty" mass. Default is ``0.0``.

        Returns
        -------
        tuple of pandas.DataFrame
            ``shap_values_df`` with raw SHAP values, ``mass_values_df`` with the
            normalized masses (including an ``"uncertainty"`` column),
            ``certainty_df`` and ``plausibility_df`` containing Dempster–Shafer
            metrics.
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

        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")

        normalized_shap = normalized_shap.mul(1 - error_rate, axis=0)
        normalized_shap["uncertainty"] = error_rate

        mass_values_df = normalized_shap.copy()

        results = []

        for idx, row in normalized_shap.iterrows():
            masses = row.drop("uncertainty").to_dict()
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

        return shap_values_df, mass_values_df, certainty_df, plausibility_df

    def ds_prompts(
        self,
        X,
        original_X,
        dataset_description,
        objective_shap,
        objective_dempster,
        top_n=3,
        error_rate=0.0,
    ):
        """Generate natural language prompts summarizing model explanations.

        This method computes the same SHAP, certainty and plausibility metrics
        as :py:meth:`ds_values` and then builds textual prompts that can be fed
        to a language model. The prompts include the original (unscaled) feature
        values and a short summary based on the most relevant features.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset for which the explanations are generated. It should contain
            the scaled features used to train the model.
        original_X : pandas.DataFrame
            Unscaled version of ``X``. The values from this dataframe are
            embedded in the generated prompts for readability.
        dataset_description : str
            A short sentence describing the dataset.
        objective_shap : str
            Text describing the objective when summarising SHAP values.
        objective_dempster : str
            Text describing the objective when summarising certainty and
            plausibility metrics.
        top_n : int, optional
            Number of top features or combinations to include in the summary.
            Defaults to ``3``.
        error_rate : float, optional
            Passed directly to :py:meth:`ds_values`.

        Returns
        -------
        tuple
            ``(shap_prompts, dempster_prompts, shap_values_df, mass_values_df,``
            ``certainty_df, plausibility_df)`` where ``*_prompts`` are
            dictionaries keyed by row index.
        """

        shap_values_df, mass_values_df, certainty_df, plausibility_df = self.ds_values(
            X, error_rate=error_rate
        )

        X_pred = self.generate_combinations(X, scaler=self.scaler)
        preds = self.model.predict(X_pred)

        for df in (shap_values_df, certainty_df, plausibility_df):
            df["prediction"] = preds

        def _get_top_features(df):
            top_dict = {}
            for idx, row in df.iterrows():
                numeric_row = row.drop(labels=["prediction"], errors="ignore")
                numeric_row = pd.to_numeric(numeric_row, errors="coerce")
                top_series = numeric_row.abs().nlargest(top_n)
                top_dict[idx] = [(col, row[col]) for col in top_series.index]
            return top_dict

        shap_top = _get_top_features(shap_values_df[original_X.columns])
        combo_cols = [c for c in certainty_df.columns if "_x_" in c]
        certainty_top = _get_top_features(certainty_df[combo_cols])
        plausibility_top = _get_top_features(plausibility_df[combo_cols])

        def resumen_shap(row_idx):
            pred = shap_values_df.loc[row_idx, "prediction"]
            shap_vals = ", ".join(name for name, _ in shap_top[row_idx])
            resumen = [
                f"Prediction for row {row_idx}: {pred}",
                f"Top SHAP features: {shap_vals}",
            ]
            return "\n".join(resumen)

        def resumen_dempster(row_idx):
            pred = shap_values_df.loc[row_idx, "prediction"]
            uncertainty = mass_values_df.loc[row_idx, "uncertainty"]
            cert_vals = ", ".join(name for name, _ in certainty_top[row_idx])
            plaus_vals = ", ".join(name for name, _ in plausibility_top[row_idx])
            resumen = [
                f"Prediction for row {row_idx}: {pred}",
                f"Uncertainty value: {uncertainty}",
                f"Certainty triples: {cert_vals}",
                f"Plausibility triples: {plaus_vals}",
            ]
            return "\n".join(resumen)

        shap_prompts = {}
        demp_prompts = {}

        for idx in shap_values_df.index:
            feature_pairs = [f"{col}={original_X.loc[idx, col]}" for col in original_X.columns]
            features_text = ", ".join(feature_pairs)

            shap_prompt = (
                dataset_description
                + f"\nObjective: {objective_shap}"
                + f"\nColumns: {features_text}\n"
                + resumen_shap(idx)
            )

            demp_prompt = (
                dataset_description
                + f"\nObjective: {objective_dempster}"
                + f"\nColumns: {features_text}\n"
                + resumen_dempster(idx)
            )

            shap_prompts[idx] = shap_prompt
            demp_prompts[idx] = demp_prompt

        return (
            shap_prompts,
            demp_prompts,
            shap_values_df,
            mass_values_df,
            certainty_df,
            plausibility_df,
        )
