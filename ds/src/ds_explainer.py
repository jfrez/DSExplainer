import warnings
import math
from collections import defaultdict
from itertools import combinations


import numpy as np
import pandas as pd


from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import shap



class DSExplainer:
    def __init__(self, model, comb, X, Y, variant="absolute"):
        self.model = model
        self.comb = comb
        self.variant = variant
        X_processed = self.generate_combinations(X)
        self.model.fit(X_processed, Y)
        self.explainer = shap.TreeExplainer(self.model)
        self.X_processed = X_processed


    def getModel(self):
        return self.model


    def generate_combinations(self, X):
        new_dataset = X.copy()
        for r in range(2, self.comb + 1):
            for cols in combinations(X.columns, r):
                new_col_name = "_x_".join(cols)
                new_dataset[new_col_name] = X[list(cols)].sum(axis=1)


        scaler = MinMaxScaler()
        new_dataset = pd.DataFrame(
            scaler.fit_transform(new_dataset),
            columns=new_dataset.columns,
            index=X.index
        )
        return new_dataset


    def ds_values(self, X, n_boot=500, alpha=0.05):
        X = self.generate_combinations(X)
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        shap_values_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)


        boot_masses = []
        for _, row in shap_values_df.iterrows():
            row_vals = row.values


            if self.variant == "absolute":
                transformed = np.abs(row_vals)
            elif self.variant == "squared":
                transformed = row_vals ** 2
            elif self.variant == "signed":
                transformed = row_vals
            elif self.variant == "normalized":
                transformed = row_vals / (np.sum(np.abs(row_vals)) + 1e-8)
            elif self.variant == "bootstrap":
                transformed = self._bootstrap_mean(row_vals, n_boot=10)
            elif self.variant == "bayes":
                transformed = self._bayes_factor(row_vals, n_boot=10)
            elif self.variant == "entropy":
                transformed = -np.abs(row_vals) * np.log(np.abs(row_vals) + 1e-8)
            else:
                raise ValueError(f"Unknown variant: {self.variant}")


            orig_sum = np.sum(np.abs(transformed))
            boot_diffs = []
            for _ in range(n_boot):
                boot_row = resample(transformed, random_state=np.random.randint(1000))
                boot_shap = np.sum(np.abs(boot_row))
                boot_diffs.append(orig_sum - boot_shap)


            ci_low, ci_high = np.percentile(
                boot_diffs, [alpha/2*100, (1-alpha/2)*100]
            )
            ci_width = max(ci_high - ci_low, 1e-8)


            feature_masses = {
                col: abs(row[col]) / (ci_width * orig_sum + 1e-8) for col in row.index
            }


            lam = 0.5
            m_theta = lam * ci_width / (lam * ci_width + orig_sum + 1e-8)
            feature_masses["THETA"] = float(m_theta)


            boot_masses.append(feature_masses)


        mass_df = pd.DataFrame(boot_masses, index=X.index)
        mass_df = mass_df.div(mass_df.sum(axis=1), axis=0).fillna(0)


        certainty_df, plausibility_df = self.compute_belief_plaus(mass_df)
        return mass_df, certainty_df, plausibility_df


    @staticmethod
    def parseset(hypname):
        return frozenset(hypname.split("_x_"))


    @staticmethod
    def _compute_belief_plaus_sets(masses_row, feature_names, thetaname="THETA"):
        focal = []
        for h in feature_names:
            m = float(masses_row.get(h, 0.0))
            if m > 0:
                focal.append((DSExplainer.parseset(h), m))


        m_theta = float(masses_row.get(thetaname, 0.0))
        if m_theta > 0:
            focal.append((None, m_theta))


        bel, pl = {}, {}
        for Aname in feature_names:
            A = DSExplainer.parseset(Aname)
            belA, plA = 0.0, 0.0
            for B, mB in focal:
                if B is None:
                    plA += mB
                else:
                    if B.issubset(A):
                        belA += mB
                    if len(B.intersection(A)) > 0:
                        plA += mB
            bel[Aname] = belA
            pl[Aname] = min(1.0, plA)
        return bel, pl


    def compute_belief_plaus(self, mass_df):
        feature_names = [c for c in mass_df.columns if c != "THETA"]


        bel_rows, pl_rows = [], []
        for _, row in mass_df.iterrows():
            masses_row = row.to_dict()
            bel, pl = self._compute_belief_plaus_sets(
                masses_row, feature_names, thetaname="THETA"
            )
            bel_rows.append(bel)
            pl_rows.append(pl)


        belief_df = pd.DataFrame(bel_rows, index=mass_df.index, columns=feature_names).fillna(0.0)
        plausibility_df = pd.DataFrame(pl_rows, index=mass_df.index, columns=feature_names).fillna(0.0)
        return belief_df, plausibility_df


    def _bootstrap_mean(self, row_vals, n_boot):
        boot_means = [np.mean(resample(row_vals)) for _ in range(n_boot)]
        return np.abs(np.array(boot_means))


    def _bayes_factor(self, row_vals, n_boot):
        boot_liks = [np.sum(np.abs(resample(row_vals))) for _ in range(n_boot)]
        bf01 = np.mean(boot_liks) / np.sum(np.abs(row_vals))
        return np.abs(row_vals) * 1/(1 + bf01)


    @staticmethod
    def _theta_from_keys(massrow: dict, thetaname="THETA"):
        atoms = set()
        for k in massrow.keys():
            if k == thetaname:
                continue
            atoms |= set(str(k).split("_x_"))
        return frozenset(atoms)


    @staticmethod
    def row_to_massdict(massrow: dict, thetaname="THETA"):
        theta = DSExplainer._theta_from_keys(massrow, thetaname=thetaname)
        m = {}
        for k, v in massrow.items():
            v = float(v)
            if v <= 0:
                continue
            if k == thetaname:
                m[theta] = v
            else:
                m[DSExplainer.parseset(k)] = v
        m.setdefault(theta, 0.0)
        return m, theta


    @staticmethod
    def massdict_to_row(m: dict, theta: frozenset, thetaname="THETA"):
        out = {}
        for S, v in m.items():
            if v <= 0:
                continue
            if S == theta:
                out[thetaname] = float(v)
            else:
                out["_x_".join(sorted(S))] = float(v)
        out.setdefault(thetaname, 0.0)
        return out


    @staticmethod
    def dempster_combine(m1: dict, m2: dict, theta: frozenset, eps: float = 1e-12):
        m1.setdefault(theta, 0.0)
        m2.setdefault(theta, 0.0)


        num = defaultdict(float)
        K = 0.0
        for A, vA in m1.items():
            if vA <= 0:
                continue
            for B, vB in m2.items():
                if vB <= 0:
                    continue
                inter = A.intersection(B)
                prod = vA * vB
                if len(inter) == 0:
                    K += prod
                else:
                    num[inter] += prod


        denom = 1.0 - K
        if denom <= eps:
            return {theta: 1.0}, K


        m12 = {S: v/denom for S, v in num.items() if v > 0}


        s = sum(m12.values())
        if s <= eps:
            m12 = {theta: 1.0}
        else:
            m12 = {S: v/s for S, v in m12.items()}


        m12.setdefault(theta, 0.0)
        return m12, K


    def combine_massdfs(self, massdf1, massdf2, thetaname="THETA"):
        rows, Ks = [], []
        for idx in massdf1.index:
            m1, theta = DSExplainer.row_to_massdict(massdf1.loc[idx].to_dict(), thetaname=thetaname)
            m2, _ = DSExplainer.row_to_massdict(massdf2.loc[idx].to_dict(), thetaname=thetaname)
            m12, K = DSExplainer.dempster_combine(m1, m2, theta)
            rows.append(DSExplainer.massdict_to_row(m12, theta, thetaname=thetaname))
            Ks.append(K)


        massdf_comb = pd.DataFrame(rows, index=massdf1.index).fillna(0.0)


        allcols = sorted(set(massdf1.columns) | set(massdf2.columns) | set(massdf_comb.columns))
        massdf_comb = massdf_comb.reindex(columns=allcols, fill_value=0.0)


        conflict = pd.Series(Ks, index=massdf1.index, name="conflict_K")
        return massdf_comb, conflict
