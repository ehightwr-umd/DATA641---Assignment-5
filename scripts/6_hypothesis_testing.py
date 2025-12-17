import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path
from .utils import load_csv

INPUT_VICTIM_CSV = Path("processed/clusters_victims.csv")
INPUT_SHOOTER_CSV = Path("processed/clusters_shooters.csv")


def chi2_test_top_clusters(df: pd.DataFrame, entity_name: str, top_n=3):
    print(f"\n=== Hypothesis Testing for {entity_name} ===")
    
    cluster_counts = df['cluster'].value_counts()
    top_clusters = cluster_counts.head(top_n).index

    for cluster_id in top_clusters:
        # Contingency table: rows=journals, columns=True/False for this cluster
        contingency = pd.crosstab(df['journal'], df['cluster'] == cluster_id)
        chi2, p, dof, expected = chi2_contingency(contingency)

        # Convert to DataFrame for easier comparison
        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

        print(f"\nCluster {cluster_id}:")
        print("Observed counts:")
        print(contingency)
        print("Expected counts if null hypothesis holds:")
        print(expected_df)

        # Calculate deviation (% difference from expected)
        deviation = (contingency - expected_df) / expected_df * 100
        print("Deviation from expected (%):")
        print(deviation.round(1))

        print(f"\nChi2={chi2:.2f}, p-value={p:.4f}")
        if p < 0.05:
            print("→ Reject null hypothesis: distribution differs across outlets")
        else:
            print("→ Fail to reject null hypothesis: distribution similar across outlets")


def main():
    df_victims = load_csv(INPUT_VICTIM_CSV)
    df_shooters = load_csv(INPUT_SHOOTER_CSV)

    if df_victims is None or df_shooters is None:
        print("Cluster files not found. Make sure 4_description_clustering was run.")
        return

    chi2_test_top_clusters(df_victims, "Victims")
    chi2_test_top_clusters(df_shooters, "Shooters")


if __name__ == "__main__":
    main()
