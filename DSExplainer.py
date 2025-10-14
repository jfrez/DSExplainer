import numpy as np
import pandas as pd
from itertools import combinations
from copy import deepcopy
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import shap

class DSExplainer:
    """
    DSExplainer con bootstrapping y fusión evidencial (regla de Dempster).

    Parámetros
    ----------
    model : estimador sklearn (árbol/ensamble compatible con shap.TreeExplainer)
    max_order : int, {1,2}. 1=solo features; 2=incluye pares si hay SHAP interactions.
    B : int, nº de réplicas bootstrap para fusión (paper sugiere 30).
    gamma : float in [0,1), masa residual m(Θ) (paper usa 0.05). Si None, usa 0.05.
    pair_percentile : float in (0,100), umbral global para filtrar pares por |ϕ_ij| medio.
    random_state : int o None
    """

    def __init__(self, model, max_order=2, B=30, gamma=0.05, pair_percentile=75.0, random_state=42):
        self.base_model = model
        self.max_order = 1 if max_order not in (1, 2) else max_order
        self.B = int(B)
        self.gamma = 0.05 if gamma is None else float(gamma)
        self.pair_percentile = float(pair_percentile)
        self.random_state = random_state

        self.scaler = MinMaxScaler()
        self.fitted_models_ = []
        self.feature_names_ = None
        self.hypotheses_ = None           # lista de frozensets({feat}) o frozensets({f1,f2})
        self.has_interactions_ = False     # True si pudimos calcular SHAP interactions
        self._preselector_model_ = None    # modelo para preselección de pares

    # ---------- Utilidades internas ----------
    def _fit_one(self, X, y, rs):
        m = clone(self.base_model)
        m.random_state = getattr(m, "random_state", rs)
        m.fit(X, y)
        return m

    def _preselect_pairs(self, model, X):
        """Usa shap_interaction_values para estimar importancia global de pares y filtrar por percentil."""
        try:
            expl = shap.TreeExplainer(model)
            inter = expl.shap_interaction_values(X)  # Para regresión: (n, d, d). Para clasificación binaria puede ser lista.
            # Normalizamos caso clasificación binaria → tomamos la clase positiva (índice 1) si viene como lista
            if isinstance(inter, list):
                inter = inter[1] if len(inter) > 1 else inter[0]
            # inter: (n,d,d). Magnitud promedio absoluta por par simétrica
            n, d, _ = inter.shape
            abs_mean = np.mean(np.abs(inter), axis=0)  # (d,d)
            # Usamos sólo la parte superior de la matriz (i<j)
            dnames = list(X.columns)
            pair_scores = {}
            for i in range(len(dnames)):
                for j in range(i+1, len(dnames)):
                    pair_scores[(dnames[i], dnames[j])] = abs_mean[i, j]
            # Umbral por percentil
            scores = np.array(list(pair_scores.values()))
            if len(scores) == 0:
                return []
            thr = np.percentile(scores, self.pair_percentile)
            selected = [frozenset(p) for p, s in pair_scores.items() if s >= thr and s > 0]
            self.has_interactions_ = len(selected) > 0
            return selected
        except Exception:
            # Si no se puede calcular interacciones, caemos a k=1
            self.has_interactions_ = False
            return []

    def _build_hypotheses(self, X, model_for_pairs):
        """Construye la familia F de hipótesis: singletons (+ pares si max_order=2 y hay interacciones)."""
        singletons = [frozenset([c]) for c in X.columns]
        if self.max_order == 1:
            return singletons
        # Intentar pares vía SHAP interactions
        pairs = self._preselect_pairs(model_for_pairs, X)
        return singletons + pairs

    def _tree_explainer(self, model):
        return shap.TreeExplainer(model)

    def _shap_singletons(self, explainer, X):
        """SHAP por feature (n,d) como DataFrame."""
        vals = explainer.shap_values(X, check_additivity=False)
        # En clasificación, TreeExplainer puede retornar lista; tomamos clase positiva si hay 2 clases
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        return pd.DataFrame(vals, columns=X.columns, index=X.index)

    def _shap_pairs(self, explainer, X):
        """SHAP interaction values (n,d,d) → DataFrame de pares con ϕ_ij (diagonal ignorada)."""
        try:
            inter = explainer.shap_interaction_values(X)
            if isinstance(inter, list):
                inter = inter[1] if len(inter) > 1 else inter[0]
            n, d, _ = inter.shape
            cols = list(X.columns)
            # Construir dict por par con valor np.array shape (n,)
            data = {}
            for i in range(d):
                for j in range(i+1, d):
                    name = f"{cols[i]}_x_{cols[j]}"
                    # Tomamos componente simétrica (i,j) (TreeSHAP ya es simétrico)
                    data[name] = inter[:, i, j]
            return pd.DataFrame(data, index=X.index)
        except Exception:
            return pd.DataFrame(index=X.index)

    def _masses_from_shap(self, shap_df, active_pairs_df, idx, gamma, hypotheses):
        """
        Construye m(H) por instancia idx:
        m(H) = |ϕ_H| / sum_A |ϕ_A| * (1 - gamma), m(Θ)=gamma.
        H son singletons y, si aplica, pares.
        """
        masses = {}
        # ϕ singletons
        phi_s = shap_df.loc[idx].to_dict()
        # ϕ pares (si los hay)
        phi_p = active_pairs_df.loc[idx].to_dict() if active_pairs_df is not None and not active_pairs_df.empty else {}

        # Preparar lista de (H, |ϕ|) sobre hipótesis activas
        terms = []
        for H in hypotheses:
            if len(H) == 1:
                (f,) = tuple(H)
                val = abs(float(phi_s.get(f, 0.0)))
                terms.append((H, val))
            elif len(H) == 2:
                f1, f2 = tuple(H)
                name = f"{f1}_x_{f2}" if f"{f1}_x_{f2}" in phi_p else f"{f2}_x_{f1}"
                val = abs(float(phi_p.get(name, 0.0)))
                terms.append((H, val))

        Z = sum(v for _, v in terms)
        if Z == 0:
            # Sin señal: asignamos toda la masa a Θ
            masses["THETA"] = 1.0
            return masses

        scale = (1.0 - gamma) / Z
        for H, v in terms:
            masses[H] = v * scale
        masses["THETA"] = gamma
        return masses

    @staticmethod
    def _intersection(A, B):
        """Intersección entre focales (frozenset de features) y/o 'THETA'."""
        if A == "THETA":
            return B
        if B == "THETA":
            return A
        inter = A.intersection(B)
        return inter if len(inter) > 0 else None  # None ≡ ∅

    @classmethod
    def _dempster_combine(cls, m1, m2):
        """
        Combina dos BPAs (diccionarios) con la regla de Dempster.
        Las claves son frozenset({...}) o 'THETA'. No se incluye ∅.
        """
        out = {}
        K = 0.0  # conflicto

        keys1 = list(m1.keys())
        keys2 = list(m2.keys())

        for A in keys1:
            for B in keys2:
                inter = cls._intersection(A, B)
                prod = m1[A] * m2[B]
                if inter is None:  # conflicto
                    K += prod
                else:
                    out[inter] = out.get(inter, 0.0) + prod

        if K >= 1.0:
            # conflicto total; devolvemos todo a THETA
            return {"THETA": 1.0}

        # Normalización por (1-K)
        norm = 1.0 - K
        out = {k: v / norm for k, v in out.items()}

        # Asegurar que no quede masa en ∅; si no hay claves, empujar a THETA
        if len(out) == 0:
            return {"THETA": 1.0}
        return out

    # ---------- API pública ----------
    def fit(self, X, y):
        """Ajusta scaler, un modelo base para preselección de pares y prepara el *bootstrap*."""
        X = X.copy()
        self.feature_names_ = list(X.columns)
        Xs = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names_, index=X.index)

        # Modelo para preseleccionar pares (si corresponde)
        self._preselector_model_ = self._fit_one(Xs, y, self.random_state)
        # Construir hipótesis (singletons + pares seleccionados si max_order=2)
        self.hypotheses_ = self._build_hypotheses(Xs, self._preselector_model_)

        # Entrenar B modelos bootstrap (sobre Xs escalado)
        rng = np.random.RandomState(self.random_state)
        self.fitted_models_ = []
        for b in range(self.B):
            idx_boot = resample(Xs.index, replace=True, random_state=rng.randint(0, 2**31 - 1))
            Xb = Xs.loc[idx_boot]
            yb = y.loc[idx_boot] if isinstance(y, pd.Series) else np.array(y)[[X.index.get_loc(i) for i in idx_boot]]
            m = self._fit_one(Xb, yb, rs=rng.randint(0, 2**31 - 1))
            self.fitted_models_.append(m)

        return self

    def ds_values(self, X):
        """
        Devuelve (shap_values_df, mass_values_df, belief_df, plausibility_df, sign_df)
        con fusión evidencial (B réplicas) y γ como masa m(Θ).

        Bel y Pl se calculan así (sin incluir m(Θ)):
        - Para singleton {i}: Bel = m({i}); Pl = m({i}) + sum m({i,j})
        - Para par {i,j}:   Bel = m({i,j}); Pl = m({i,j}) + m({i}) + m({j})
        """
        assert self.hypotheses_ is not None, "Debes llamar .fit(X,y) primero."

        Xs = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names_, index=X.index)

        # Para cada réplica: SHAP singletons y (si aplica) pares
        shap_singletons_list = []
        shap_pairs_list = []
        for m in self.fitted_models_:
            expl = self._tree_explainer(m)
            s_df = self._shap_singletons(expl, Xs)
            shap_singletons_list.append(s_df)
            if self.max_order == 2 and self.has_interactions_:
                p_df = self._shap_pairs(expl, Xs)
            else:
                p_df = pd.DataFrame(index=Xs.index)
            shap_pairs_list.append(p_df)

        # Signo por hipótesis (mediana de ϕ_H a través de réplicas)
        # Singletons
        phi_s_stack = np.stack([df.values for df in shap_singletons_list], axis=0)  # (B, n, d)
        phi_s_median = np.median(phi_s_stack, axis=0)  # (n, d)
        sign_singletons = np.sign(phi_s_median)
        sign_df = pd.DataFrame(sign_singletons, columns=self.feature_names_, index=X.index)

        # Pares
        pair_names = set()
        for df in shap_pairs_list:
            pair_names |= set(df.columns)
        pair_names = sorted(list(pair_names))
        if pair_names:
            # rellenar faltantes con 0 antes de apilar
            pairs_filled = []
            for df in shap_pairs_list:
                df2 = df.reindex(columns=pair_names, fill_value=0.0)
                pairs_filled.append(df2)
            phi_p_stack = np.stack([df.values for df in pairs_filled], axis=0)  # (B, n, n_pairs)
            phi_p_median = np.median(phi_p_stack, axis=0)  # (n, n_pairs)
            sign_pairs = pd.DataFrame(np.sign(phi_p_median), columns=pair_names, index=X.index)
        else:
            sign_pairs = pd.DataFrame(index=X.index)

        # Fusión evidencial por instancia (combina B masas con Dempster)
        fused_masses_per_row = {}
        for idx in X.index:
            # construir masas por réplica y combinar
            m_star = None
            for b in range(self.B):
                s_df = shap_singletons_list[b]
                p_df = shap_pairs_list[b]
                masses_b = self._masses_from_shap(s_df, p_df, idx, self.gamma, self.hypotheses_)
                if m_star is None:
                    m_star = masses_b
                else:
                    m_star = self._dempster_combine(m_star, masses_b)
            fused_masses_per_row[idx] = m_star

        # Convertir masas fusionadas a DataFrame (columnas = hipótesis + 'THETA')
        cols = []
        for H in self.hypotheses_:
            if len(H) == 1:
                (f,) = tuple(H)
                cols.append(f)
            elif len(H) == 2:
                f1, f2 = tuple(H)
                cols.append(f"{f1}_x_{f2}")
        cols_plus_theta = cols + ["uncertainty"]  # mostramos m(THETA) como 'uncertainty'

        mass_records = []
        for idx, m in fused_masses_per_row.items():
            row = {}
            # mapear cada H a nombre de columna
            for H in self.hypotheses_:
                if len(H) == 1:
                    (f,) = tuple(H)
                    row[f] = m.get(H, 0.0)
                elif len(H) == 2:
                    f1, f2 = tuple(H)
                    row[f"{f1}_x_{f2}"] = m.get(H, 0.0)
            row["uncertainty"] = m.get("THETA", 0.0)
            row["_index"] = idx
            mass_records.append(row)

        mass_values_df = pd.DataFrame(mass_records).set_index("_index").reindex(columns=cols_plus_theta).fillna(0.0)

        # Bel y Pl (sin incluir 'uncertainty' en Pl)
        # Reglas:
        #  - Bel({i}) = m({i})
        #  - Pl({i})  = m({i}) + sum_{j!=i} m({i}_x_{j})
        #  - Bel({i,j}) = m({i}_x_{j})
        #  - Pl({i,j})  = m({i}_x_{j}) + m({i}) + m({j})
        # Construimos columnas
        bel = {}
        pl = {}

        # Índices auxiliares
        single_cols = self.feature_names_
        pair_cols = [c for c in mass_values_df.columns if "_x_" in c]

        # Bel/Pl singletons
        for f in single_cols:
            bel[f] = mass_values_df[f]
            related_pairs = [c for c in pair_cols if c.startswith(f + "_x_") or c.endswith("_x_" + f)]
            pl[f] = mass_values_df[f] + mass_values_df[related_pairs].sum(axis=1) if related_pairs else mass_values_df[f]

        # Bel/Pl pares
        for c in pair_cols:
            f1, f2 = c.split("_x_")
            bel[c] = mass_values_df[c]
            pl[c] = mass_values_df[c] + mass_values_df[f1] + mass_values_df[f2]

        belief_df = pd.DataFrame(bel, index=mass_values_df.index)
        plausibility_df = pd.DataFrame(pl, index=mass_values_df.index)

        # SHAP (mediana) para reportar valores firmados (útil en análisis local)
        shap_values_df = pd.DataFrame(phi_s_median, columns=self.feature_names_, index=X.index)
        if pair_names:
            shap_values_df = pd.concat([shap_values_df, pd.DataFrame(phi_p_median, columns=pair_names, index=X.index)], axis=1)

        # Agregar signos (separados) como referencia
        sign_full = sign_df.copy()
        if not sign_pairs.empty:
            sign_full = pd.concat([sign_full, sign_pairs], axis=1)

        return shap_values_df, mass_values_df, belief_df, plausibility_df, sign_full

    def ds_prompts(self, X, original_X, dataset_description, objective_shap, objective_dempster, top_n=3):
        """
        Igual que antes, pero usando masas fusionadas y Bel/Pl resultantes.
        Devuelve (shap_prompts, dempster_prompts, shap_values_df, mass_values_df, belief_df, plausibility_df)
        """
        shap_values_df, mass_values_df, belief_df, plausibility_df, sign_df = self.ds_values(X)

        # Predicciones del ensamble: promedio de predicciones de los B modelos
        Xs = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names_, index=X.index)
        preds = np.mean([m.predict(Xs) for m in self.fitted_models_], axis=0)

        # Top-N por |ϕ| (solo columnas originales para SHAP-top)
        shap_top = {}
        base_cols = self.feature_names_
        for i, idx in enumerate(X.index):
            row = shap_values_df.loc[idx, base_cols]
            top_series = row.abs().nlargest(top_n)
            shap_top[idx] = [(col, row[col]) for col in top_series.index]

        # Top-N por Bel/Pl en combos (pares)
        combo_cols = [c for c in belief_df.columns if "_x_" in c]
        cert_top = {}
        plaus_top = {}
        for idx in X.index:
            cert_series = belief_df.loc[idx, combo_cols].nlargest(top_n) if combo_cols else pd.Series(dtype=float)
            plaus_series = plausibility_df.loc[idx, combo_cols].nlargest(top_n) if combo_cols else pd.Series(dtype=float)
            cert_top[idx] = list(cert_series.items())
            plaus_top[idx] = list(plaus_series.items())

        def resumen_shap(row_idx):
            pred = preds[list(X.index).index(row_idx)]
            shap_names = ", ".join(name for name, _ in shap_top[row_idx])
            return f"Prediction for row {row_idx}: {pred}\nTop SHAP features: {shap_names}"

        def resumen_dempster(row_idx):
            pred = preds[list(X.index).index(row_idx)]
            unc = mass_values_df.loc[row_idx, "uncertainty"] * 100.0
            cert_vals = ", ".join(f"{k}: {v*100:.2f}%" for k, v in cert_top[row_idx])
            plaus_vals = ", ".join(f"{k}: {v*100:.2f}%" for k, v in plaus_top[row_idx])
            return (
                f"Prediction for row {row_idx}: {pred}\n"
                f"Uncertainty value: {unc:.2f}%\n"
                f"Certainty (Bel) top: {cert_vals}\n"
                f"Plausibility (Pl) top: {plaus_vals}"
            )

        shap_prompts = {}
        demp_prompts = {}
        for idx in X.index:
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

        return shap_prompts, demp_prompts, shap_values_df, mass_values_df, belief_df, plausibility_df
