## Libraries & Imports:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency
from sentence_transformers import SentenceTransformer
import umap

## Paths
INPUT_VICTIM_CSV = Path("processed/clusters_victims.csv")
INPUT_SHOOTER_CSV = Path("processed/clusters_shooters.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


## Load CSVs
def load_data():
    df_victims = pd.read_csv(INPUT_VICTIM_CSV)
    df_shooters = pd.read_csv(INPUT_SHOOTER_CSV)
    df_victims['entity'] = 'victim'
    df_shooters['entity'] = 'shooter'
    df = pd.concat([df_victims, df_shooters], ignore_index=True)
    return df, df_victims, df_shooters

## Frequency and Proportion Heatmaps
def plot_cluster_heatmap(df):
    freq_table = pd.crosstab([df['journal'], df['entity']], df['cluster'])
    prop_table = pd.crosstab([df['journal'], df['entity']], df['cluster'], normalize='index')

    plt.figure(figsize=(12, 8))
    sns.heatmap(prop_table, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Proportion'})
    plt.title("Proportion of Description Clusters Across Outlets and Entities")
    plt.xlabel("Cluster ID")
    plt.ylabel("Outlet / Entity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_proportion_heatmap.png")
    plt.close()

## Bar Charts for Top N Clusters
def plot_top_clusters(df, top_n=5):
    top_clusters = df['cluster'].value_counts().head(top_n).index
    for cluster_id in top_clusters:
        cluster_data = df[df['cluster'] == cluster_id].groupby(['journal', 'entity']).size().unstack(fill_value=0)
        ax = cluster_data.plot(kind='bar', figsize=(10,6), stacked=True, colormap='tab20')
        plt.title(f"Cluster {cluster_id} Frequency by Outlet and Entity")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cluster_{cluster_id}_bar.png")
        plt.close()

## Chi-Squared Deviation Plots
def chi2_deviation_plot(df, entity_name, top_n=3):
    cluster_counts = df['cluster'].value_counts()
    top_clusters = cluster_counts.head(top_n).index

    for cluster_id in top_clusters:
        contingency = pd.crosstab(df['journal'], df['cluster'] == cluster_id)
        chi2, p, dof, expected = chi2_contingency(contingency)
        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
        deviation = (contingency - expected_df) / expected_df * 100

        deviation_true = deviation.get(True, deviation.iloc[:,0])

        plt.figure(figsize=(8,5))
        deviation_true.plot(kind='bar', color='coral')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"{entity_name} Cluster {cluster_id} Deviation from Expected (%)")
        plt.ylabel("Deviation (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{entity_name.lower()}_cluster_{cluster_id}_deviation.png")
        plt.close()

        print(f"{entity_name} Cluster {cluster_id} Chi2={chi2:.2f}, p-value={p:.4f}")

## UMAP Visualization

def plot_umap(df):
    df['descriptions'] = df.apply(
        lambda row: row.get('victim_descriptions') if row['entity']=='victim' else row.get('shooter_descriptions'), axis=1
    )
    descriptions = df['descriptions'].fillna('').tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions, show_progress_bar=True)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x=embedding_2d[:,0],
        y=embedding_2d[:,1],
        hue=df['cluster'],
        style=df['entity'],
        palette='tab20',
        s=50,
        alpha=0.8
    )
    plt.title("UMAP Projection of Descriptions by Cluster and Entity")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_clusters.png")
    plt.close()

## Main Pipeline

def main():
    df, df_victims, df_shooters = load_data()
    plot_cluster_heatmap(df)
    plot_top_clusters(df)
    chi2_deviation_plot(df_victims, "Victims")
    chi2_deviation_plot(df_shooters, "Shooters")
    plot_umap(df)
    print(f"All figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
