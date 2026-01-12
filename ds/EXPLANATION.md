# DSExplainer (`ds_explainer.py`) — Function-by-function explanation (Dempster–Shafer)

This module defines a single class, `DSExplainer`, that turns per-instance SHAP feature attributions into **Dempster–Shafer Theory (DST)** objects: a basic probability assignment (mass function) `m(·)`, plus **belief** (Bel) and **plausibility** (Pl), and it also supports evidence combination using **Dempster’s rule**.
## Class: `DSExplainer`

### `getModel(self)`
- Returns the trained model stored in `self.model`. 

### `generate_combinations(self, X)`
- Copies `X` and creates additional columns for all combinations of features of size `r = 2..comb`, using names like `"feat1_x_feat2"`. 
- Each new column is computed as the row-wise sum of the participating original columns. 
- Applies `MinMaxScaler` to scale every column to a common range and returns the scaled `DataFrame`. 

### `ds_values(self, X, n_boot=500, alpha=0.05)`
- Regenerates the interaction-augmented feature matrix for the input `X`. 
- Computes SHAP values with the tree explainer and stores them in a `DataFrame` aligned with columns. 
- For each instance (row), transforms SHAP values according to `variant`:
  - `absolute`: 
    uses `abs(SHAP)`. 
  - `squared`: 
    uses `SHAP^2`. 
  - `signed`: 
    uses raw signed SHAP values. 
  - `normalized`: 
    divides by the sum of absolute SHAP values (with a small epsilon). 
  - `bootstrap`: 
    uses `_bootstrap_mean` as an alternative transformation. 
  - `bayes`: 
    uses `_bayes_factor` as a shrinkage-like transformation. 
  - `entropy`: 
    uses `-|x| log(|x|)` (with epsilon). 
- Uses bootstrap resampling of the transformed vector to compute a percentile interval width `ci_width` from `boot_diffs`, which serves as a stability/uncertainty proxy. 
- Builds a per-feature mass-like score and a special mass `THETA`:
  - Feature masses are scaled by `1 / (ci_width * orig_sum + eps)`. 
  - `THETA` mass (`m_theta`) depends on `ci_width` and `orig_sum` (with `lam=0.5`), and represents residual ignorance. 
- Normalizes each row so masses sum to 1, then computes belief and plausibility with `compute_belief_plaus`. 
- Returns `(mass_df, certainty_df, plausibility_df)`. 

### `parseset(hypname)` *(static)*
- Converts a column name like `"A_x_B"` into `frozenset({"A","B"})`. 

### `_compute_belief_plaus_sets(masses_row, feature_names, thetaname="THETA")` *(static)*
- Builds the list of focal elements from non-zero masses; `THETA` is handled as a special entry (`None`) that contributes to plausibility. 
- For each hypothesis `A`:
  - Belief accumulates mass of focal sets `B` such that `B ⊆ A`. 
  - Plausibility accumulates mass of focal sets `B` such that `B ∩ A ≠ ∅`, and also adds `THETA` mass because Θ does not exclude `A`. 
- Returns dictionaries `bel` and `pl`. 

### `compute_belief_plaus(self, mass_df)`
- Iterates over each row of `mass_df`, calls `_compute_belief_plaus_sets`, and assembles `belief_df` and `plausibility_df`. 

### `_bootstrap_mean(self, row_vals, n_boot)`
- Resamples `row_vals` `n_boot` times and takes the mean of each resample; returns absolute values of those means. 

### `_bayes_factor(self, row_vals, n_boot)`
- Resamples `row_vals` and uses the sum of absolute values as a proxy “likelihood”; forms `bf01` as a ratio of bootstrap mean to the original sum and shrinks `abs(row_vals)` by `1/(1+bf01)`. 

### `_theta_from_keys(massrow: dict, thetaname="THETA")` *(static)*
- Collects all atomic feature names appearing in the keys (splitting by `"_x_"`) and returns them as a `frozenset`, representing Θ. 

### `row_to_massdict(massrow: dict, thetaname="THETA")` *(static)*
- Converts a flat row dict (string keys) into a DST-style mass dictionary whose keys are sets (`frozenset`). 
- Maps `THETA` to the computed `theta` set and ensures `theta` exists in the output. 

### `massdict_to_row(m: dict, theta: frozenset, thetaname="THETA")` *(static)*
- Converts a set-keyed mass dictionary back into a flat dict with keys like `"A_x_B"` and `THETA`. 

### `dempster_combine(m1: dict, m2: dict, theta: frozenset, eps: float = 1e-12)` *(static)*
- Implements Dempster’s rule: multiplies masses for all pairs `(A,B)`, accumulates product mass on `A ∩ B`, and accumulates **conflict** `K` when `A ∩ B = ∅`. 
- Normalizes by `1-K`; if `1-K` is too small (near-total conflict), returns full mass on Θ as a fallback and still returns `K`. 


### `combine_massdfs(self, massdf1, massdf2, thetaname="THETA")`
- For each row index, converts both rows to set-keyed mass dicts, combines them with `dempster_combine`, converts back to a flat row, and builds a combined `DataFrame`. 
- Returns `(massdf_comb, conflict_series)` where `conflict_series` contains per-row conflict values `K`. 

# Data Preparation (`data_prep.py`) — Function-by-function explanation

This module loads the “No-show appointments” Kaggle dataset and applies a simple preprocessing pipeline (dropping columns, removing missing values, label-encoding all remaining columns, and splitting into train/test). 

## Function: `load_kaggle_dataset(file_path, dataset=)`

### What it does
- Downloads (and locally caches) a Kaggle dataset via `kagglehub.dataset_download(dataset)` and then loads a specific file into a `pandas.DataFrame` using `kagglehub.dataset_load` with `KaggleDatasetAdapter.PANDAS`. 

### Inputs
- `file_path`: the path/name of the file inside the Kaggle dataset to load (default: `KaggleV2-May-2016.csv`). 
- `dataset`: Kaggle dataset identifier (default: `joniarroba/noshowappointments`). 

### Outputs
- Returns a `pandas.DataFrame` containing the loaded CSV. 

### Notes
- The variable `path` (result of `dataset_download`) is not used afterward; the code relies on `dataset_load` to read the file. 

## Function: `preprocess_split(df, target_column, drop_columns, test_size=0.1, random_state=42)`

### What it does
- Cleans and encodes the raw dataframe, then returns a standard scikit-learn `train_test_split` output: `X_train, X_test, y_train, y_test`. 

### Step-by-step behavior
- Copies the input `df` to avoid mutating it in place. 
- Drops `drop_columns` (by default identifiers and date/time fields). 
- Drops rows with missing values using `dropna()`. 
- Builds `ordinal_features`, a mapping from each column name to the list of unique observed values (`data[col].unique().tolist()`). 
- For **every** remaining column (including the target column), creates a `LabelEncoder`, fits it on the observed categories list, and transforms the column to integer codes. 
- Separates the encoded target `target = data[target_column]` and defines `features = data.drop(columns=[target_column])`. 
- Calls `train_test_split(X, y, test_size=test_size, random_state=random_state)` and returns its four outputs. 

### Important implementation notes
- `LabelEncoder` is primarily intended for encoding target labels rather than input features; for categorical input features, `OrdinalEncoder` or one-hot encoding is often preferred depending on the model. 
- Fitting an encoder separately **per column** is fine, but fitting it on `data[col].unique()` means the integer mapping depends on the unique values present in `df` and their order; for reproducibility across different dataset slices, it is common to `fit` encoders on the training split only and then `transform` the test split with the same fitted encoders. 
- `MinMaxScaler` is imported but not used in this file. 

# Experiments (`experiments.py`) — Function-by-function explanation

This module orchestrates experiments around `DSExplainer` outputs (mass / belief / plausibility) and SHAP bootstrap baselines, producing summary metrics, sweeps over the interaction order `k`, correlation reports, and stability analyses. 

## Function: `evaluate_one_setting(Xtrain, ytrain, Xeval, model, k=3, variant="absolute", n_boot=200, alpha=0.05)`

### What it does
- Runs one experimental configuration (one `k` and one `variant`) and returns a single dictionary (a “row”) of metrics combining DST-style outputs from `DSExplainer` and a SHAP-bootstrap baseline width. 

### Step-by-step behavior
- Clones the `model` (to avoid contaminating the caller’s estimator) and constructs a `DSExplainer` with `comb=k` and the requested `variant`, fitting it on `(Xtrain, ytrain)`. 
- Calls `expl.ds_values(Xeval, n_boot=n_boot, alpha=alpha)` to obtain `massdf`, `beldf`, `pldf` for the evaluation set. 
- Computes aggregate DST metrics with `summarize_dsexplainer_outputs(massdf, beldf, pldf)` (defined in the local `.metrics` module). 
- Generates the same interaction-augmented feature matrix for `Xeval` via `expl.generate_combinations(Xeval)` and computes SHAP bootstrap intervals with `shap_bootstrap_intervals`. 
- Defines baseline metrics from the SHAP bootstrap interval widths: mean width and median width over all cells in `width`. 
- Returns a dictionary containing `k`, `variant`, `nboot`, the DST metrics, and the SHAP baseline metrics. 

## Function: `theoretical_num_hypotheses(p, k)`

### What it does
- Computes the theoretical number of hypotheses created when including all combinations of `p` original features up to order `k`: 
  \(\sum_{r=1}^{k} \binom{p}{r}\). 

## Function: `realized_num_hypotheses(explainer, X_sample)`

### What it does
- Calls `explainer.generate_combinations(X_sample)` and returns the number of resulting columns (`Xk.shape[1]`). 

## Function: `run_k_sweep(Xtrain, ytrain, Xeval, model, variant="absolute", k_list=(1, 2, 3, 4), n_boot=200)`

### What it does
- Runs a sweep over multiple `k` values and returns a `DataFrame` summarizing how the DST outputs change as the hypothesis space grows. 

### Step-by-step behavior
- For each `k` in `k_list`, builds a fresh `DSExplainer` with `comb=k` and computes `(mass, bel, pl)` via `ds_values`. 
- Computes:
  - `theta_mean`: mean mass assigned to `THETA` (if present). 
  - `belpl_width`: average of `(pl - bel)` (mean width across all instances and hypotheses). 
  - `n_hyp`: number of non-Θ hypotheses (`mass.shape[1] - 1` if Θ exists). 
- Returns these per-`k` rows as a `DataFrame`. 


## Function: `correlation_report(X, method="pearson", thr=0.85)`

### What it does
- Computes the absolute correlation matrix `|corr(X)|` and returns all feature pairs whose absolute correlation is at least `thr`. 

### Step-by-step behavior
- Forms `C = X.corr(...).abs()`, then scans the upper triangle and collects pairs `(feat1, feat2, abs_corr)` meeting the threshold. 
- Returns a `DataFrame` sorted by `abs_corr` descending. 


## Function: `top_hypotheses_from_mass(mass_row, top_n=10)`

### What it does
- Extracts the top `top_n` hypothesis names (columns) by mass from a single mass row, excluding `THETA`. 


## Function: `jaccard(a, b)`

### What it does
- Computes Jaccard similarity between two collections: `|A ∩ B| / |A ∪ B|` (with a `max(1, ...)` guard). 

## Function: `stability_bootstrap(Xtrain, ytrain, x0, model, variant="absolute", k=3, B=25, top_n=10)`

### What it does
- Measures how stable the *top-mass hypotheses* are for a single target instance `x0`, when the training data is bootstrapped `B` times. 

### Step-by-step behavior
- Repeats `B` times: resamples `(Xtrain, ytrain)`, trains a fresh `DSExplainer`, runs `ds_values(x0, ...)`, and records the top hypotheses from `mass.iloc[0]`. 
- Computes all pairwise Jaccard similarities between the `B` top-hypothesis lists and returns their mean. 


## Function: `top_row_to_records(df, variant, metric, row_index, class_label, top_n)`

### What it does
- Turns the top `top_n` values of a specific row in a metric `DataFrame` into a list of record dicts (for later concatenation into a table). 

### Step-by-step behavior
- Selects `row = df.iloc[row_index]`, takes `row.nlargest(top_n)`, and emits dicts with `variant`, `metric`, `row_index`, `class_label`, `rank`, `feature`, and `value`. 

## Function: `analyze_variants(Xtrain, ytrain, Xtest, ytest, model, variants, max_combinations=3, top_n=3)`

### What it does
- For each `variant`, runs `DSExplainer` on up to two representative test instances (one from each class if available) and returns a table of top-ranked hypotheses for mass, certainty (belief), and plausibility. 

### Step-by-step behavior
- Builds masks for class 0 and class 1 and selects up to one instance from each, concatenating them into `X_combined` (or falls back to `Xtest[:2]`). 
- For each `variant`, constructs a `DSExplainer` with `comb=max_combinations` and runs `ds_values(X_combined)`. 
- Separates out the `THETA` mass (if present) and drops `THETA` from the per-hypothesis tables for ranking. 
- For each of the first two rows, builds records:
  - Adds an explicit record for `THETA` with rank 0 (if present). 
  - Adds top `top_n` hypotheses for `mass`, `certainty` (belief), and `plausibility` using `top_row_to_records`. 
- Returns `{"top_df": top_df}` where `top_df` is a `DataFrame` of all these records. 

# Combination Experiments (`experiments_combination.py`) — Function-by-function explanation

This module runs a focused experiment that builds two `DSExplainer` instances (with different SHAP-to-evidence transformations) and then **combines** their mass functions using Dempster’s rule, reporting conflict and Θ-mass statistics.

## Function: `run_abs_sq_combination(model, Xtrain, ytrain, Xeval, k=3, n_boot=200, outdir=None, prefix="abs_sq")`

### What it does
- Trains two `DSExplainer` objects on the same training data and model family, one using `variant="absolute"` and one using `variant="squared"`. 
- Computes their mass/belief/plausibility outputs on `Xeval`, then combines the two mass functions per instance using `combine_massdfs` (Dempster’s rule implementation inside `DSExplainer`). 
- Computes belief and plausibility for the combined mass function and summarizes key quantities (conflict and mean Θ mass) and optionally writes CSV outputs. 

### Inputs
- `model`: a scikit-learn compatible estimator; it is cloned so the two explainers train independent copies. 
- `Xtrain`, `ytrain`: training set used inside each `DSExplainer` to fit the model on interaction-augmented features. 
- `Xeval`: evaluation set on which masses and Bel/Pl are computed. 
- `k`: maximum interaction order passed as `comb` into `DSExplainer`. 
- `n_boot`: bootstrap repetitions passed to `ds_values` for mass construction. 
- `outdir`: if not `None`, output directory where CSVs are written. 
- `prefix`: filename prefix used for outputs. 

### Step-by-step behavior
- Builds `expl_abs = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant="absolute")` and runs `ds_values` to get `(mass_abs, bel_abs, pl_abs)` on `Xeval`. 
- Builds `expl_sq = DSExplainer(clone(model), comb=k, X=Xtrain, Y=ytrain, variant="squared")` and runs `ds_values` to get `(mass_sq, bel_sq, pl_sq)` on `Xeval`. 
- Combines the two per-instance mass functions with `mass_comb, conflictK = expl_abs.combine_massdfs(mass_abs, mass_sq)`. 
- Computes belief and plausibility for the combined masses with `bel_comb, pl_comb = expl_abs.compute_belief_plaus(mass_comb)`. 
- Builds a `summary` dictionary with:
  - `k`, `n_eval`, `n_boot`. 
  - `conflictK_mean`: average conflict across instances. 
  - `THETA_abs_mean`, `THETA_sq_mean`, `THETA_comb_mean`: average Θ mass for each source and after combination. 
- If `outdir` is provided:
  - Creates the directory and writes `summary` to `{prefix}_summary.csv`. 
  - Writes the three mass tables to `{prefix}_mass_absolute.csv`, `{prefix}_mass_squared.csv`, and `{prefix}_mass_combined.csv`. 
- Returns a dict containing the summary, the individual outputs for both variants, the combined outputs, and the per-instance conflict series. 

# Metrics (`metrics.py`) — Function-by-function explanation

This module defines small helpers to summarize `DSExplainer` outputs (especially Θ mass and the Bel/Pl interval width) and to format “top values” from a selected row for display/logging.

## Function: `summarize_dsexplainer_outputs(massdf, beldf, pldf)`

### What it does
- Produces a compact dictionary of summary statistics from three `DataFrame`s returned by `DSExplainer.ds_values`: the mass table (`massdf`), belief table (`beldf`), and plausibility table (`pldf`).

### Step-by-step behavior
- Initializes an output dict `out = {}`.
- If the mass table has a `"THETA"` column:
  - Computes `theta_mean` as the mean of `massdf["THETA"]`.
  - Computes `theta_median` as the median of `massdf["THETA"]`.
- Otherwise, sets `theta_mean` and `theta_median` to `np.nan`.
- Finds the set of comparable hypothesis columns for belief/plausibility:
  - `common_cols = [c for c in beldf.columns if c in pldf.columns and c != "THETA"]`.
- Computes the “belief–plausibility interval widths” for those columns:
  - `widths = (pldf[common_cols] - beldf[common_cols]).values`.
- Aggregates widths into:
  - `belpl_width_mean`: overall mean of all entries in `widths`.
  - `belpl_width_median`: overall median of all entries in `widths`.
- Returns the dictionary `out`.

## Function: `format_top_row(df, df_name, row_index, top_n)`

### What it does
- Builds a human-readable string listing the top `top_n` largest values in a chosen row of a `DataFrame`.

### Step-by-step behavior
- Selects the row by position: `row = df.iloc[row_index]`.
- Picks the `top_n` largest entries with `row.nlargest(top_n)`.
- Constructs a list of lines starting with a header like `"{df_name}, Row {row_index}:"`, then one line per `(column, value)` pair.
- Returns the final multi-line string joined with newline characters.

# `shap_baselines.py` — Function-by-function explanation 

This module provides one utility function, `shap_bootstrap_intervals`, that computes per-instance **bootstrap intervals** for SHAP attributions (optionally after a simple transformation such as absolute value or squaring). 

## Function: `shap_bootstrap_intervals(explainer_obj, X, n_boot=200, alpha=0.05, variant="absolute", random_state=0)`

### Purpose
- Computes, for each row in `X`, bootstrap-based summary statistics of SHAP values **across features**: mean, lower/upper percentile bounds, and interval width. 

### Parameters
- `explainer_obj`: expected to have an attribute `explainer` that exposes `shap_values(...)` (for example, a wrapper like the `DSExplainer` class that stores `self.explainer`). 
- `X`: a `pandas.DataFrame` of input features for which SHAP values will be computed. 
- `n_boot`: number of bootstrap resamples per instance. 
- `alpha`: significance level; the returned interval is the percentile interval 
  \([\alpha/2, 1-\alpha/2]\). 
- `variant`: transformation applied to a row’s SHAP vector before bootstrapping (`absolute`, `squared`, `signed`, `normalized`, `entropy`). 
- `random_state`: seed used to make bootstrap sampling reproducible. 

### Step-by-step behavior
1. Initializes a NumPy RNG with `random_state`. 
2. Calls `explainer_obj.explainer.shap_values(X, check_additivity=False)` to compute SHAP values. 
3. Handles SHAP return types:
   - If SHAP returns a list with one element, it uses that single array. 
   - If SHAP returns a list with multiple arrays, it tries to concatenate them horizontally (`np.hstack`). 
   - If concatenation fails due to shape mismatch, it emits a warning and falls back to the first array only. 
   - If SHAP returns a single array, it uses it directly. 
4. Aligns column names:
   - If the SHAP array has more columns than `X`, it appends placeholder names (`feature_i`). 
   - If it has fewer, it truncates `X.columns` to match. 
   - Otherwise it uses `X.columns` as-is. 
5. Builds `shap_df`, a DataFrame of SHAP values with the finalized columns and the same index as `X`. 
6. Defines `transform(row_vals)` to apply one of the supported transformations:
   - `absolute`: `abs(row_vals)`. 
   - `squared`: `row_vals**2`. 
   - `signed`: identity. 
   - `normalized`: divides by sum of absolute SHAP (with epsilon). 
   - `entropy`: `-|x| log(|x|)` (with epsilon). 
7. For each instance (each row in `shap_df`):
   - Applies the selected transform. 
   - Creates `n_boot` bootstrap samples by resampling the transformed vector *with replacement*. 
   - Stacks bootstraps into a matrix `boots` of shape `(n_boot, n_features)` and computes per-feature percentiles:
     - `lo = percentile(alpha/2)` and `hi = percentile(1-alpha/2)`. 
   - Also computes the per-feature bootstrap mean and interval width `(hi - lo)`. 
8. Returns four DataFrames, each shaped `(n_rows, n_features)`:
   - `mean`: bootstrap mean per feature and instance. 
   - `low`: lower percentile bound per feature and instance. 
   - `high`: upper percentile bound per feature and instance. 
   - `width`: interval width (`high-low`) per feature and instance. 
