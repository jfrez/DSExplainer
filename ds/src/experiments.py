import math
import numpy as np
import pandas as pd


from sklearn.base import clone
from sklearn.utils import resample


from .ds_explainer import DSExplainer
from .metrics import summarize_dsexplainer_outputs, format_top_row
from .shap_baselines import shap_bootstrap_intervals



def evaluate_one_setting(Xtrain, ytrain, Xeval, model, k=3, variant="absolute", n_boot=200, alpha=0.05):
    expl = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant=variant)
    massdf, beldf, pldf = expl.ds_values(Xeval, n_boot=n_boot, alpha=alpha)
    ds_metrics = summarize_dsexplainer_outputs(massdf, beldf, pldf)


    Xeval_k = expl.generate_combinations(Xeval)
    mean_shap, low, high, width = shap_bootstrap_intervals(expl, Xeval_k, n_boot=n_boot, alpha=alpha, variant=variant)


    base_metrics = {
        "shap_boot_width_mean": float(width.values.mean()),
        "shap_boot_width_median": float(np.median(width.values)),
    }


    row = {"k": k, "variant": variant, "nboot": n_boot}
    row.update(ds_metrics)
    row.update(base_metrics)
    return row



def theoretical_num_hypotheses(p, k):
    return sum(math.comb(p, r) for r in range(1, k + 1))



def realized_num_hypotheses(explainer, X_sample):
    Xk = explainer.generate_combinations(X_sample)
    return Xk.shape[1]



def run_k_sweep(Xtrain, ytrain, Xeval, model, variant="absolute", k_list=(1, 2, 3, 4), n_boot=200):
    rows = []
    for k in k_list:
        expl = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant=variant)
        mass, bel, pl = expl.ds_values(Xeval, n_boot=n_boot)
        theta_mean = mass["THETA"].mean() if "THETA" in mass.columns else np.nan
        belpl_width = (pl - bel).mean().mean()
        n_hyp = mass.shape[1] - (1 if "THETA" in mass.columns else 0)
        rows.append({
            "k": int(k),
            "variant": variant,
            "n_hypotheses": int(n_hyp),
            "theta_mean": float(theta_mean),
            "mean_bel_pl_width": float(belpl_width),
        })
    return pd.DataFrame(rows)



def correlation_report(X, method="pearson", thr=0.85):
    C = X.corr(method=method).abs()
    pairs = []
    cols = list(C.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if C.iloc[i, j] >= thr:
                pairs.append([cols[i], cols[j], float(C.iloc[i, j])])
    return pd.DataFrame(pairs, columns=["feat1", "feat2", "abs_corr"]).sort_values("abs_corr", ascending=False)



def top_hypotheses_from_mass(mass_row, top_n=10):
    s = mass_row.drop(labels=["THETA"], errors="ignore")
    return list(s.sort_values(ascending=False).head(top_n).index)



def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))



def stability_bootstrap(Xtrain, ytrain, x0, model, variant="absolute", k=3, B=25, top_n=10):
    top_sets = []
    for b in range(B):
        Xb, yb = resample(Xtrain, ytrain, replace=True, random_state=1000 + b)
        expl = DSExplainer(clone(model), comb=k, X=Xb, Y=yb, variant=variant)
        mass, bel, pl = expl.ds_values(x0, n_boot=200)
        top_sets.append(top_hypotheses_from_mass(mass.iloc[0], top_n=top_n))


    sims = []
    for i in range(B):
        for j in range(i + 1, B):
            sims.append(jaccard(top_sets[i], top_sets[j]))
    return float(np.mean(sims))


def top_row_to_records(df, variant, metric, row_index, class_label, top_n):
    row = df.iloc[row_index]
    top_values = row.nlargest(top_n)

    records = []
    for rank, (feature, value) in enumerate(top_values.items(), start=1):
        records.append({
            "variant": variant,
            "metric": metric,
            "row_index": row_index,
            "class_label": class_label,
            "rank": rank,
            "feature": feature,
            "value": value
        })
    return records

def analyze_variants(Xtrain, ytrain, Xtest, ytest, model, variants, max_combinations=3, top_n=3):
    class_0_mask = ytest == 0
    class_1_mask = ytest == 1

    top_records = []


    for variant in variants:
        explainer = DSExplainer(model, comb=max_combinations, X=Xtrain, Y=ytrain, variant=variant)


        X_noshow = Xtest[class_0_mask][:1]
        X_show = Xtest[class_1_mask][:1]
        X_combined = pd.concat([X_noshow, X_show]) if (len(X_noshow) > 0 and len(X_show) > 0) else Xtest[:2]


        mass_df, certainty_df, plausibility_df = explainer.ds_values(X_combined)


        theta_mass = mass_df["THETA"].copy() if "THETA" in mass_df.columns else None
        mass_df_no_theta = mass_df.drop(columns=["THETA"], errors="ignore")
        certainty_df_no_theta = certainty_df.drop(columns=["THETA"], errors="ignore")
        plausibility_df_no_theta = plausibility_df.drop(columns=["THETA"], errors="ignore")


        for i in range(min(2, len(mass_df))):
            class_label = "NO" if i == 0 else "YES"

            if theta_mass is not None:
                top_records.append({
                    "variant": variant,
                    "metric": "mass",
                    "row_index": i,
                    "class_label": class_label,
                    "rank": 0,
                    "feature": "THETA",
                    "value": float(theta_mass.iloc[i])
                })

            top_records += top_row_to_records(mass_df_no_theta, variant, "mass", i, class_label, top_n)
            top_records += top_row_to_records(certainty_df_no_theta, variant, "certainty", i, class_label, top_n)
            top_records += top_row_to_records(plausibility_df_no_theta, variant, "plausibility", i, class_label, top_n)
    
    top_df = pd.DataFrame.from_dict(top_records)

    return {"top_df": top_df}    