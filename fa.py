'''
In this file, the Exploratory Factor Analysis 
is excecuted based on the output excel file from descriptive_statistics.py
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

EXCEL_FILE = "results/FA_res.xlsx"
DATA_FIELDS = [
    "Number of Nodes", "Number of Levels",
    "Connector Heterogeneity", "Branching Factor", "Path Complexity", "Gate Diversity"
]
def run_factor_analysis(csv_path: str, n_factors: int = None, compute_clusters: bool = False):
    df = pd.read_excel(csv_path)
    df = df[DATA_FIELDS].dropna()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # KMO and Bartlett
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    kmo_series = pd.Series(kmo_all, index=DATA_FIELDS)
    print("KMO values per variable:")
    print(kmo_series.sort_values())

    # Check for KMO scores
    low_kmo = kmo_series[kmo_series < 0.50]
    if not low_kmo.empty:
        print("\nVariables with low KMO (< 0.50):")
        print(low_kmo)
    
    chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled)

    print(f"KMO Measure: {kmo_model:.4f}")
    print(f"Bartlett's Test: chi2={chi_square_value:.2f}, p={p_value:.4f}")

    if n_factors is None:
        fa = FactorAnalyzer(rotation=None)
        fa.fit(data_scaled)
        ev, _ = fa.get_eigenvalues()

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(ev)+1), ev, marker='o')
        plt.axhline(1, color='r', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        plt.show()

        print("Use the scree plot to determine number of factors.")
        return

    # Fit final model with Varimax rotation
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(data_scaled)

    communalities = pd.Series(fa.get_communalities(), index=DATA_FIELDS).clip(upper=0.9999)
    factor_scores = fa.transform(data_scaled)

    var_exp = fa.get_factor_variance()
    print("Variance Explained by Factors:")
    print(pd.DataFrame({
        'Factor': [f"F{i+1}" for i in range(n_factors)],
        'SS Loadings': var_exp[0],
        'Proportion Var': var_exp[1],
        'Cumulative Var': var_exp[2]
    }))

    # Factor Loadings
    loadings = pd.DataFrame(fa.loadings_, index=DATA_FIELDS, columns=[f"F{i+1}" for i in range(n_factors)])
    print("\nFactor Loadings:")
    print(loadings.round(3))

    if compute_clusters:
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(factor_scores)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(factor_scores)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='Set2', s=100)
        plt.title("Clusters based on Factor Scores (PCA Reduced)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()

    return loadings.round(3), communalities, factor_scores

if __name__ == "__main__":
    run_factor_analysis("results/ft_metrics.xlsx")
    fl, communalities, scores = run_factor_analysis("results/ft_metrics.xlsx", n_factors=2, compute_clusters=True)  # Step 2: Clusters
    with pd.ExcelWriter(EXCEL_FILE) as writer:
        fl.to_excel(writer, sheet_name="fa_results")
        communalities.to_frame("Communality").to_excel(writer, sheet_name="communalities")
        print(communalities)
        pd.DataFrame(scores, columns=[f"F{i+1}" for i in range(scores.shape[1])]).to_excel(writer, sheet_name="factor_scores")
