import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import load_csv, save_csv

INPUT_VICTIM_CSV = Path("processed/clusters_victims.csv")
INPUT_SHOOTER_CSV = Path("processed/clusters_shooters.csv")

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    df_victims = load_csv(INPUT_VICTIM_CSV)
    df_shooters = load_csv(INPUT_SHOOTER_CSV)

    if df_victims is None or df_shooters is None:
        print("Clusters files not found.")
        return

    df_victims['entity'] = 'victim'
    df_shooters['entity'] = 'shooter'

    df = pd.concat([df_victims, df_shooters], ignore_index=True)

    df['cluster'] = df['cluster'].astype(int)
    df = df.sort_values('cluster')
    df['cluster'] = df['cluster'].astype(str)

    freq_table = pd.crosstab([df['journal'], df['entity']], df['cluster'])
    freq_table.to_csv(OUTPUT_DIR / "cluster_frequency.csv")
    print("Saved cluster frequency table.")

    prop_table = pd.crosstab([df['journal'], df['entity']], df['cluster'], normalize='index')
    prop_table.to_csv(OUTPUT_DIR / "cluster_proportion.csv")
    print("Saved cluster proportion table.")

    for cluster_id in sorted(df['cluster'].unique()):
        print(f"\nCluster {cluster_id}: sample sentences")
        sample_sentences = df[df['cluster'] == cluster_id]['sentence'].head(5).tolist()
        for s in sample_sentences:
            print(f" - {s}")

if __name__ == "__main__":
    main()
