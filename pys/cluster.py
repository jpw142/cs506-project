import sys
import json
import numpy as np
import umap
import hdbscan

def load_embeddings(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ids = list(data.keys())
    embeddings = np.array([data[id_] for id_ in ids])
    return ids, embeddings

def reduce_embeddings(embeddings, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(embeddings)

def cluster_embeddings(embeddings):
    return hdbscan.HDBSCAN(
      min_cluster_size=5, 
      min_samples=2, 
      prediction_data=True
    ).fit(embeddings)

def save_clusters(ids, labels, output_json):
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({id_: int(lbl) for id_, lbl in zip(ids, labels)}, f, indent=2)
    print(f"Clusters saved to {output_json}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python clustering.py <input_json> <dim> <output_json>")
        sys.exit(1)

    input_file, dim, output_json = sys.argv[1:]
    ids, embs = load_embeddings(input_file)

    if embs.shape[1] != int(dim):
        print(f"Expected {dim} dimensions, got {embs.shape[1]}")
        sys.exit(1)

    cl = cluster_embeddings(embs)
    save_clusters(ids, cl.labels_, output_json)

if __name__ == "__main__":
    main()

