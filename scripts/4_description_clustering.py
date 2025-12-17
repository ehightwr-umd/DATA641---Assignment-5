## Libraries & Imports:
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from .utils import timeit, save_csv, load_csv
import numpy as np

## Paths:

INPUT_VICTIM_CSV = Path("processed/descriptions_victims.csv")
INPUT_SHOOTER_CSV = Path("processed/descriptions_shooters.csv")
OUTPUT_VICTIM_CLUSTER_CSV = Path("processed/clusters_victims.csv")
OUTPUT_SHOOTER_CLUSTER_CSV = Path("processed/clusters_shooters.csv")

## Load Model: Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

## Function Utilities:
def preprocess_descriptions(df, col):
    df = df.copy()
    df['descriptions'] = df[col].fillna('').str.split(' \| ')
    df = df.explode('descriptions')
    df['descriptions'] = df['descriptions'].str.lower().str.strip()
    df['descriptions'] = df['descriptions'].str.replace(r'[^\w\s]', '', regex=True)
    df = df[df['descriptions'].str.len() > 2]  # remove too-short phrases
    df = df.drop_duplicates(subset=['descriptions', 'sentence'])
    return df

## Function to Embed Text
@timeit
def embed_text(sentences):
    clean_sentences = [str(s) for s in sentences if isinstance(s, str) and s.strip()]
    embeddings = model.encode(clean_sentences, show_progress_bar=True)
    embeddings = normalize(embeddings)  # normalize for cosine distance
    return embeddings, clean_sentences

## Function for Dimensionality Reduction:
@timeit
def reduce_dimensions(embeddings, n_components=15):
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    return reducer.fit_transform(embeddings)

@timeit
## Functions for Clustering:
def cluster_embeddings(embeddings_2d, min_cluster_size=3, cluster_selection_epsilon=0.05):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(embeddings_2d)
    return labels

def merge_similar_clusters(df, embeddings, labels, threshold=0.85):
    """
    Merge clusters whose centroids have cosine similarity above threshold
    """
    df['cluster'] = labels
    unique_labels = np.unique(labels)
    centroids = []

    for lbl in unique_labels:
        if lbl == -1:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroids.append(embeddings[labels == lbl].mean(axis=0))
    centroids = np.vstack(centroids)
    sim_matrix = cosine_similarity(centroids)

    label_map = {lbl: lbl for lbl in unique_labels}
    for i, lbl_i in enumerate(unique_labels):
        if lbl_i == -1:
            continue
        for j, lbl_j in enumerate(unique_labels):
            if i >= j or lbl_j == -1:
                continue
            if sim_matrix[i, j] > threshold:
                # merge j into i
                label_map[lbl_j] = lbl_i

    df['cluster'] = df['cluster'].map(label_map)
    return df


## Process & Cluster:
def process_file(input_csv: Path, output_csv: Path, entity_name: str):
    df = load_csv(input_csv)
    if df is None:
        print(f"{entity_name} input CSV not found: {input_csv}")
        return

    column_map = {"Victims": "victim_descriptions", "Shooters": "shooter_descriptions"}
    col = column_map[entity_name]

    df = preprocess_descriptions(df, col)

    if df.empty:
        print(f"No descriptions to cluster in {input_csv}")
        return

    embeddings, _ = embed_text(df['descriptions'].tolist())
    embeddings_2d = reduce_dimensions(embeddings, n_components=15)
    labels = cluster_embeddings(embeddings_2d, min_cluster_size=3, cluster_selection_epsilon=0.05)
    df = merge_similar_clusters(df, embeddings, labels, threshold=0.85)

    save_csv(df, output_csv)
    print(f"Saved {entity_name} clusters: {output_csv}")


## Main Pipeline:
def main():
    process_file(INPUT_VICTIM_CSV, OUTPUT_VICTIM_CLUSTER_CSV, "Victims")
    process_file(INPUT_SHOOTER_CSV, OUTPUT_SHOOTER_CLUSTER_CSV, "Shooters")

if __name__ == "__main__":
    main()
