import sys
import csv
import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/specter2_base"

print("Loading tokenizer and model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed_text(text):
    text = text or ""
    with torch.no_grad():
        inputs = tokenizer([text], padding=True, truncation=True,
                           return_tensors="pt", max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def load_cache(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_titles(path):
    lookup = {}
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = (row.get('NoticeId') or '').strip()
            title = (row.get('Title') or '').strip()
            if nid:
                lookup[nid] = title
    return lookup

class SemanticSearch:
    def __init__(self, cache_json_path, titles_csv_path):
        import os 
        print("Cache path:", os.path.abspath(cache_json_path))
        print("Loading embeddings cache…")
        self.cache = load_cache(cache_json_path)
        print("ids")
        self.ids = list(self.cache.keys())
        print("mat")
        self.mat = np.array([self.cache[k] for k in self.ids])

        print("Loading titles lookup…")
        self.title_lookup = load_titles(titles_csv_path)

    def query(self, sentence, threshold=0.5, top_k=None):
        print("Embedding input sentence…")
        sent_emb = embed_text(sentence)

        print("Calculating cosine similarities…")
        sims = cosine_similarity(sent_emb, self.mat)[0]

        results = []
        for nid, sim in zip(self.ids, sims):
            if sim >= threshold:
                results.append((nid, float(sim), self.title_lookup.get(nid, "")))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            results = results[:top_k]

        return results

def save_results(results, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["NoticeId", "CosineSimilarity", "Title"])
        for nid, sim, title in results:
            writer.writerow([nid, sim, title])
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py \"your sentence here\" cache.json titles.csv threshold output.csv")
        sys.exit(1)

    _, sentence, cache_path, titles_path, threshold_str, out_csv = sys.argv

    searcher = SemanticSearch(cache_path, titles_path)
    results = searcher.query(sentence, threshold=float(threshold_str))
    print(f"Found {len(results)} matches >= threshold {threshold_str}")
    save_results(results, out_csv)

