import numpy as np
import pandas as pd
from itertools import combinations
from copy import deepcopy
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import shap

import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import shap


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
        """Fit the estimator and prepare the SHAP explainer."""
        self.model = model
        self.comb = int(comb)
        self.scaler = MinMaxScaler()

        Xc = self.generate_combinations(X, scaler=self.scaler)
        model.fit(Xc, Y)
        self.explainer = shap.TreeExplainer(model)

    def getModel(self):
        """Return the underlying fitted model."""
        return self.model

    def generate_combinations(self, X, scaler=None):
        """Create scaled columns for feature combinations (optimized, no fragmentation)."""
        new_dataset = X.copy()

        # Generar todas las combinaciones y concatenarlas en un solo paso
        combo_frames = []
        for r in range(2, self.comb + 1):
            combo_frames.append(
                pd.concat(
                    [
                        pd.Series(X[list(cols)].sum(axis=1), name="_x_".join(cols))
                        for cols in combinations(X.columns, r)
                    ],
                    axis=1
                )
            )

        if combo_frames:
            new_dataset = pd.concat([new_dataset] + combo_frames, axis=1)

        # Escalado
        scaler = scaler or self.scaler
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


    def _parse_hypothesis(self, name):
        """Return set of atomic features contained in a hypothesis name."""
        return set(name.split("_x_"))

    def _build_focal_sets(self, cols):
        """List of hypothesis names (all original features + all generated combos)."""
        # Exclude 'uncertainty' if present (it isn't among cols at this point)
        return [c for c in cols]

    def ds_values(self, X, error_rate=0.0):
        """Compute SHAP masses, Belief ('certainty') and Plausibility for X.

        Notes
        -----
        - Masses are proportional to |SHAP| per hypothesis and normalized to (1 - error_rate).
        - m(THETA) = error_rate is stored in the 'uncertainty' column.
        - certainty_df implements Belief; plausibility_df implements Plausibility.
        """
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")

        # Ensure identical transformation pipeline used at fit time
        Xc = self.generate_combinations(X, scaler=self.scaler)

        # SHAP values on the *same* columns used to train
        shap_vals = self.explainer.shap_values(Xc, check_additivity=False)
        # TreeExplainer can return list for classification; select array when needed
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        shap_values_df = pd.DataFrame(shap_vals, columns=Xc.columns, index=Xc.index)

        # Build masses from absolute SHAP
        focal_cols = self._build_focal_sets(shap_values_df.columns)
        abs_shap = shap_values_df[focal_cols].abs()
        denom = abs_shap.sum(axis=1).replace(0.0, 1e-12)  # avoid division by zero
        masses = abs_shap.div(denom, axis=0) * (1.0 - error_rate)
        masses["uncertainty"] = error_rate
        mass_values_df = masses.copy()

        # Prepare Belief (certainty) and Plausibility
        # Map hypothesis -> atomic feature set
        hypo_atoms = {h: self._parse_hypothesis(h) for h in focal_cols}

        bel_rows = []
        pl_rows = []
        for idx, row in masses.iterrows():
            # exclude THETA from DS sums
            m = row.drop(labels=["uncertainty"]).to_dict()

            bel_row = {}
            pl_row = {}

            for h_name, h_atoms in hypo_atoms.items():
                # Belief: sum of masses of all subsets A ⊆ H
                bel = 0.0
                for a_name, a_atoms in hypo_atoms.items():
                    if a_atoms.issubset(h_atoms):
                        bel += m.get(a_name, 0.0)
                bel_row[h_name] = bel

                # Plausibility: sum of masses of all A with A ∩ H ≠ ∅
                pl = 0.0
                for a_name, a_atoms in hypo_atoms.items():
                    if h_atoms.intersection(a_atoms):
                        pl += m.get(a_name, 0.0)
                pl_row[h_name] = pl

            bel_row["uncertainty"] = row["uncertainty"]
            pl_row["uncertainty"] = row["uncertainty"]
            bel_rows.append(bel_row)
            pl_rows.append(pl_row)

        certainty_df = pd.DataFrame(bel_rows, index=masses.index)
        plausibility_df = pd.DataFrame(pl_rows, index=masses.index)

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
        """Generate natural language prompts summarizing model explanations."""
        shap_values_df, mass_values_df, certainty_df, plausibility_df = self.ds_values(
            X, error_rate=error_rate
        )

        X_pred = self.generate_combinations(X, scaler=self.scaler)
        preds = self.model.predict(X_pred)

        for df in (shap_values_df, certainty_df, plausibility_df):
            df["prediction"] = preds

        def _get_top(df):
            out = {}
            for idx, row in df.iterrows():
                numeric = pd.to_numeric(
                    row.drop(labels=["prediction"], errors="ignore"), errors="coerce"
                )
                top = numeric.abs().nlargest(top_n)
                out[idx] = [(col, row[col]) for col in top.index]
            return out

        # Only original atomic features for the SHAP top list
        original_cols = [c for c in original_X.columns if c in shap_values_df.columns]
        shap_top = _get_top(shap_values_df[original_cols])

        combo_cols = [c for c in certainty_df.columns if "_x_" in c]
        cert_top = _get_top(certainty_df[combo_cols]) if combo_cols else {}
        pl_top = _get_top(plausibility_df[combo_cols]) if combo_cols else {}

        def _summary_shap(i):
            pred = shap_values_df.loc[i, "prediction"]
            names = ", ".join(name for name, _ in shap_top.get(i, []))
            return f"Prediction for row {i}: {pred}\nTop SHAP features: {names}"

        def _summary_ds(i):
            pred = shap_values_df.loc[i, "prediction"]
            unc = mass_values_df.loc[i, "uncertainty"] * 100
            cert_vals = ", ".join(f"{k}: {v * 100:.2f}%" for k, v in cert_top.get(i, []))
            pl_vals = ", ".join(f"{k}: {v * 100:.2f}%" for k, v in pl_top.get(i, []))
            return (
                f"Prediction for row {i}: {pred}\n"
                f"Uncertainty value: {unc:.2f}%\n"
                f"Certainty (Belief) top: {cert_vals}\n"
                f"Plausibility top: {pl_vals}"
            )

        shap_prompts, demp_prompts = {}, {}
        for i in shap_values_df.index:
            feats = ", ".join(f"{col}={original_X.loc[i, col]}" for col in original_X.columns)
            shap_prompts[i] = (
                dataset_description
                + f"\nObjective: {objective_shap}"
                + f"\nColumns: {feats}\n"
                + _summary_shap(i)
            )
            demp_prompts[i] = (
                dataset_description
                + f"\nObjective: {objective_dempster}"
                + f"\nColumns: {feats}\n"
                + _summary_ds(i)
            )

        return (
            shap_prompts,
            demp_prompts,
            shap_values_df,
            mass_values_df,
            certainty_df,
            plausibility_df,
        )
