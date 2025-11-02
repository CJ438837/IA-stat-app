import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, probplot
from fitter import Fitter
import streamlit as st
from io import BytesIO

def advanced_distribution_analysis(df, types_df, output_folder=None):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import shapiro, kstest, probplot
    from fitter import Fitter
    import pandas as pd

    results = []
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()

    for col in num_vars:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        # Test de normalité
        if len(col_data) < 5000:
            stat, p_value = shapiro(col_data)
            test_used = "Shapiro-Wilk"
        else:
            stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
            test_used = "Kolmogorov-Smirnov"
        verdict = "Normal" if p_value > 0.05 else "Non Normal"

        # Détection distribution probable
        try:
            if np.all(col_data == col_data.astype(int)):
                f = Fitter(col_data, distributions=['poisson','binom'])
            else:
                f = Fitter(col_data, distributions=['norm','expon','lognorm','uniform'])
            f.fit()
            summary_df = f.summary()
            best_fit = summary_df['Distribution'].iloc[0] if not summary_df.empty else "unknown"
        except:
            best_fit = "unknown"

        results.append({
            "variable": col,
            "n": len(col_data),
            "normality_test": test_used,
            "statistic": stat,
            "p_value": p_value,
            "verdict": verdict,
            "best_fit_distribution": best_fit
        })

        # Graphiques dans Streamlit
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.histplot(col_data, kde=True, color='skyblue', bins=20)
        plt.title(f"{col} - Histogramme + KDE")
        plt.xlabel(col)
        plt.ylabel("Fréquence")

        plt.subplot(1,2,2)
        probplot(col_data, dist="norm", plot=plt)
        plt.title(f"{col} - QQ Plot")
        plt.tight_layout()

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(f"{output_folder}/{col}_distribution.png")
        # Affiche dans Streamlit
        st.pyplot(plt.gcf())
        plt.close()

    return pd.DataFrame(results)

