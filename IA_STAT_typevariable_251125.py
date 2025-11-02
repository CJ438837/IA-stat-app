def detect_variable_types_df(df):
    import pandas as pd
    import numpy as np

    results = []
    for col in df.columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        unique_vals = pd.Series(col_data).astype(str).str.strip().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            var_type = "binaire"
        elif np.issubdtype(col_data.dtype, np.number):
            var_type = "numérique"
        else:
            var_type = "catégorielle"

        results.append({
            "variable": col,
            "type": var_type,
            "valeurs_uniques": n_unique,
            "exemples": unique_vals[:5]
        })

    types_df = pd.DataFrame(results)
    return {"data": types_df}, {"data": df}
