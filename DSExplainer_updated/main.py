import os
import warnings
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.data_prep import load_kaggle_dataset, preprocess_split
from src.ds_explainer import DSExplainer
from src.experiments import (
    run_k_sweep,
    evaluate_one_setting,
    theoretical_num_hypotheses,
    realized_num_hypotheses,
    analyze_variants,
    correlation_report,
)
from src.experiments_combination import run_abs_sq_combination

def main():
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    OUTDIR = "no-show"
    os.makedirs(OUTDIR, exist_ok=True)

    df = load_kaggle_dataset(file_path="KaggleV2-May-2016.csv", dataset="joniarroba/noshowappointments")
    Xtrain, Xtest, ytrain, ytest = preprocess_split(
        df, target_column="No-show", drop_columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], test_size=0.1, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    p = Xtrain.shape[1]
    k_list_hspace = [1, 2, 3, 4]

    
    rows = []
    for k in k_list_hspace:
        tmp = DSExplainer(
            RandomForestRegressor(n_estimators=100, random_state=42),
            comb=k,
            X=Xtrain,
            Y=ytrain,
            variant="absolute",
        )
        realized = realized_num_hypotheses(tmp, Xtrain.iloc[:5])
        rows.append(
            {
                "k": int(k),
                "p_original": int(p),
                "theoretical_|H_k|": int(theoretical_num_hypotheses(p, k)),
                "realized_num_columns": int(realized),
            }
        )

    df_hspace = pd.DataFrame(rows)
    df_hspace.to_csv(os.path.join(OUTDIR, "hypothesis_space_size.csv"), index=False)

    k_list = [1, 2, 3]
    df_k = run_k_sweep(
        Xtrain, ytrain,
        Xeval=Xtest.iloc[:200],
        model=model,
        variant="absolute",
        k_list=k_list,
        n_boot=200
    )
    df_k.to_csv(os.path.join(OUTDIR, "k_sweep_summary.csv"), index=False)

    corr_pairs = correlation_report(Xtrain, thr=0.05)
    corr_pairs.to_csv(os.path.join(OUTDIR, "correlation_report_premodel.csv"), index=False)

    variants = ['absolute', 'squared', 'signed', 'normalized', 'bootstrap', 'bayes', 'entropy']
    results = analyze_variants(
        Xtrain=Xtrain,
        ytrain=ytrain,
        Xtest=Xtest,
        ytest=ytest,
        model=model,
        variants=variants,
        max_combinations=3,
        top_n=3
    )

    results["top_df"].to_csv(os.path.join(OUTDIR, "ds_top_values.csv"), index=False)

    variants = ['absolute', 'squared', 'signed', 'normalized', 'entropy']

    X_eval = Xtest.iloc[:200]
    rows = []
    for k in k_list:
        for v in variants:
            row = evaluate_one_setting(
                Xtrain=Xtrain,
                ytrain=ytrain,
                Xeval=X_eval,
                model=model,
                k=k,
                variant=v,
                n_boot=200,
                alpha=0.05,
            )
            rows.append(row)

    df_results = pd.DataFrame(rows).sort_values(["k", "variant"])
    df_results.to_csv(os.path.join(OUTDIR, "results_baselines.csv"), index=False)

    print("Saved CSVs in:", OUTDIR)
    print(df_hspace.head())
    print(df_k.head())
    print(df_results.head())

    combo = run_abs_sq_combination(
        model=model,
        Xtrain=Xtrain,
        ytrain=ytrain,
        Xeval=Xtest.iloc[:50],
        k=3,
        n_boot=200,
        outdir=OUTDIR,
        prefix="abs_sq_k3"
    )

    print("Average conflict K:", combo["summary"]["conflictK_mean"])
    print("Average THETA (combined):", combo["summary"]["THETA_comb_mean"])

if __name__ == "__main__":
    main()