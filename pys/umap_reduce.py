import json
import os
import sys
import numpy as np
import umap
import joblib

def load_json(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    
    # Convert dict of ID -> vector into just a list of vectors
    ids = list(raw.keys())
    vectors = np.array(list(raw.values()))
    
    return vectors, ids

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

# Input:
# Json hashmap of NoticeID -> Semantic Embedding
# Dimensions of Semantic Embedding
# Where to save output json to
# what the desired dimension of semantic embedding in output json
# The model path to either save to or load from
#
# Reduces the dimensionality of data with UMAP Algorithm
def main():
    if len(sys.argv) != 6:
        print("Usage: python umap_reduce.py input.json dimension_in output.json dimension_out")
        sys.exit(1)

    input_path = sys.argv[1]
    dim_in = int(sys.argv[2])
    output_path = sys.argv[3]
    dim_out = int(sys.argv[4])
    model_path = sys.argv[5]

    print(f"Loading data from {input_path}...")
    data, ids = load_json(input_path)

    if data.shape[1] != dim_in:
        print(f"Error: Data has dimension {data.shape[1]} but expected {dim_in}")
        sys.exit(1)

    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing UMAP model from {model_path}...")
        reducer = joblib.load(model_path)
    else:
        print("Creating new UMAP model...")
        reducer = umap.UMAP(n_components=dim_out, metric="cosine")
        reducer.fit(data)
        joblib.dump(reducer, model_path)
        print(f"Model saved to {model_path}")

    print("Transforming data...")
    reduced = reducer.transform(data)

    print(f"Saving reduced data to {output_path}...")
    output_dict = {id_: vector.tolist() for id_, vector in zip(ids, reduced)}
    save_json(output_path, output_dict)
    print("Done.")

if __name__ == "__main__":
    main()
