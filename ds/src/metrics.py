import numpy as np

def summarize_dsexplainer_outputs(massdf, beldf, pldf):
    out = {}
    if "THETA" in massdf.columns:
        out["theta_mean"] = float(massdf["THETA"].mean())
        out["theta_median"] = float(massdf["THETA"].median())
    else:
        out["theta_mean"] = np.nan
        out["theta_median"] = np.nan


    common_cols = [c for c in beldf.columns if c in pldf.columns and c != "THETA"]
    widths = (pldf[common_cols] - beldf[common_cols]).values
    out["belpl_width_mean"] = float(np.mean(widths))
    out["belpl_width_median"] = float(np.median(widths))
    return out



def format_top_row(df, df_name, row_index, top_n):
    row = df.iloc[row_index]
    top_values = row.nlargest(top_n)
    lines = [f"{df_name}, Row {row_index}:"]
    for col, val in top_values.items():
        lines.append(f" {col}: {val}")
    return "\n".join(lines)
