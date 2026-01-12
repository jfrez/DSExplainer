import os
import pandas as pd
from sklearn.base import clone

from src.ds_explainer import DSExplainer


def run_abs_sq_combination(
    model,
    Xtrain,
    ytrain,
    Xeval,
    k=3,
    n_boot=200,
    outdir=None,
    prefix="abs_sq",
):
    expl_abs = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant="absolute")
    mass_abs, bel_abs, pl_abs = expl_abs.ds_values(Xeval, n_boot=n_boot)

    expl_sq = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant="squared")
    mass_sq, bel_sq, pl_sq = expl_sq.ds_values(Xeval, n_boot=n_boot)

    mass_comb, conflictK = expl_abs.combine_massdfs(mass_abs, mass_sq)
    bel_comb, pl_comb = expl_abs.compute_belief_plaus(mass_comb)

    summary = {
        "k": int(k),
        "n_eval": int(len(Xeval)),
        "n_boot": int(n_boot),
        "conflictK_mean": float(getattr(conflictK, "mean")()),
        "THETA_abs_mean": float(mass_abs["THETA"].mean()),
        "THETA_sq_mean": float(mass_sq["THETA"].mean()),
        "THETA_comb_mean": float(mass_comb["THETA"].mean()),
    }

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame([summary]).to_csv(os.path.join(outdir, f"{prefix}_summary.csv"), index=False)
        mass_abs.to_csv(os.path.join(outdir, f"{prefix}_mass_absolute.csv"), index=False)
        mass_sq.to_csv(os.path.join(outdir, f"{prefix}_mass_squared.csv"), index=False)
        mass_comb.to_csv(os.path.join(outdir, f"{prefix}_mass_combined.csv"), index=False)

    return {
        "summary": summary,
        "mass_abs": mass_abs, "bel_abs": bel_abs, "pl_abs": pl_abs,
        "mass_sq": mass_sq, "bel_sq": bel_sq, "pl_sq": pl_sq,
        "mass_comb": mass_comb, "bel_comb": bel_comb, "pl_comb": pl_comb,
        "conflictK": conflictK,
    }