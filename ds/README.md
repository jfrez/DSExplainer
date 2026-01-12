# What it does and how to run it

This script is the runnable entry point of the project: it downloads the Kaggle “No-show appointments” dataset, preprocesses it, trains a baseline model, runs several `DSExplainer`/DST-style experiments, and saves multiple CSV reports under the `no-show/` folder (can be modified.). 

## How to execute

### 1) Project layout expectation
- The script imports modules as `from src...`, so it expects a package/folder named `src/` containing:
  - `data_prep.py` 
  - `ds_explainer.py`
  - `experiments.py` 
  - `experiments_combination.py` 
  - `metrics.py` 
  - `shap_baselines.py` 

### 2) Install dependencies
- Install the project dependencies from the existing `requirements.txt`:

```bash
pip install -r requirements.txt
```

- Make sure the file includes the direct imports used by this project 

### 3) Kaggle access
- `load_kaggle_dataset(...)` uses `kagglehub` to download/load `dataset` (default: `joniarroba/noshowappointments`) and the file `file_path` 
(default: `KaggleV2-May-2016.csv`). 
- Ensure Kaggle credentials are configured for `kagglehub` in your environment; otherwise dataset download will fail. 

### 4) Run
From the project root (so that `src/` is importable), run:

```bash
python main.py
```

The script writes its outputs into the configured output directory (created if missing).

## What `main()` does (step-by-step)

### Setup
- Creates an output directory `OUTDIR` (default: `"no-show"`). 

### Data loading and preprocessing
- Downloads/loads the dataset and calls `preprocess_split(...)` to produce `Xtrain, Xtest, ytrain, ytest`. 
- The default preprocessing drops ID/date columns and label-encodes all remaining columns (including the target). 

### Baseline model
- Instantiates `RandomForestRegressor(n_estimators=100, random_state=42)`. 

### Hypothesis-space size report
- For `k_list_hspace = [1,2,3,4]`, it creates a temporary `DSExplainer` and compares:
  - Theoretical number of hypotheses \(\sum_{r=1}^{k} \binom{p}{r}\) via `theoretical_num_hypotheses(p,k)`. 
  - Realized number of columns after `generate_combinations` via `realized_num_hypotheses(...)`. 
- Saves `hypothesis_space_size.csv` into `OUTDIR/`. 

### k-sweep report
- Runs `run_k_sweep(...)` for `k_list = [1,2,3]` on `Xtest.iloc[:200]` using `variant="absolute"` and `n_boot=200`. 
- Saves `k_sweep_summary.csv` into `OUTDIR/`. 

### Correlation report
- Computes `correlation_report(Xtrain, thr=0.05)` and saves `correlation_report_premodel.csv`. 

### Per-variant “top hypotheses” table
- Defines a broader variant list: `['absolute','squared','signed','normalized','bootstrap','bayes','entropy']`. 
- Calls `analyze_variants(...)` (max combinations `=3`, top `n=3`) and saves `ds_top_values.csv`. 

### Baselines comparison table
- Defines a reduced variant list for the baseline table: `['absolute','squared','signed','normalized','entropy']`. 
- For each `k in [1,2,3]` and each variant, calls `evaluate_one_setting(...)` on `Xtest.iloc[:200]`. 
- Saves the combined table as `results_baselines.csv` and prints a few `.head()` previews to stdout. 

### Evidence-combination experiment
- Calls `run_abs_sq_combination(...)` on `Xtest.iloc[:50]` with `k=3`, `n_boot=200`, writing outputs with prefix `abs_sq_k3`. 
- Prints average conflict `K` and combined Θ mean mass to stdout. 

## Outputs produced
All files are written under `OUTDIR/` by default: 
- `hypothesis_space_size.csv` 
- `k_sweep_summary.csv` 
- `correlation_report_premodel.csv` 
- `ds_top_values.csv` 
- `results_baselines.csv` 
- `abs_sq_k3_summary.csv`, `abs_sq_k3_mass_absolute.csv`, `abs_sq_k3_mass_squared.csv`, `abs_sq_k3_mass_combined.csv` (only from the combination run). 
