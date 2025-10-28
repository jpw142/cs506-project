import sys
import csv
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ProcessPoolExecutor

# ---------------------------
# CONFIG
# "541511",
# "541512",
# "541990"
# "541690",
# ---------------------------
RELEVANT_NAICS = {
    "541715" 
}

MODEL_NAME = "allenai/specter2_base"
BOILERPLATE_PATH = "boilerplate_phrases.csv"

print("Loading tokenizer and model globally…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ---------------------------
# CLEANING UTILITIES
# ---------------------------
def load_boilerplate(path):
    with open(path, newline='', encoding='utf-8') as f:
        return set(row[0].strip().lower() for row in csv.reader(f) if row)

def clean_contract_text(title, description, boilerplate_phrases):
    text = f"{title} {tokenizer.sep_token} {description}".lower()
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, '')
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    cleaned = text.strip()
    count = len(cleaned)
    return cleaned, count

# ---------------------------
# EMBEDDING FUNCTION
# ---------------------------
def embed_chunk(texts):
    texts = [(t if t is not None else '') for t in texts]
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True,
                           return_tensors="pt", max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ---------------------------
# HELPERS
# ---------------------------
def load_cache(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No or invalid cache found—starting fresh.")
        return {}

def save_json(obj, path, desc="Saved"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"{desc} → {path}")

def load_filtered_opps(csv_path):
    filtered = []
    seen_titles = set()
    with open(csv_path, newline='', encoding='utf-8', errors='ignore') as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            naics = (row.get('NaicsCode') or '').strip()
            title = (row.get('Title') or '').strip()
            if naics in RELEVANT_NAICS and title and title not in seen_titles:
                seen_titles.add(title)
                clean_row = {k: (v if v is not None else '') for k, v in row.items()}
                filtered.append(clean_row)
    print(f"Kept {len(filtered)} unique-title opportunities")
    return filtered

def load_capabilities(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [(line.strip() if line is not None else '') for line in f if line.strip()]

def parallel_embed(texts, batch_size=16, workers=4, desc="Embedding"):
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    all_embs = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for emb in tqdm(executor.map(embed_chunk, batches),
                        total=len(batches), desc=desc, unit="batch"):
            all_embs.append(emb)
    return np.vstack(all_embs)

def embed_missing(opps, cache, boilerplate_phrases):                                             
    missing = []
    texts = []                                                                
    for row in opps:                                                                                                                 
        if row.get('NoticeId') not in cache:                             
            cleaned, word_count = clean_contract_text(row.get('Title', ''), row.get('Description', ''), boilerplate_phrases)
            if word_count >= 10:
                missing.append(row)                                            
                texts.append(cleaned)                                                                                                   
    if not missing:                                                                                 
        print("No new descriptions to embed.")                                                                       
        return cache                                                                                
    embs = parallel_embed(texts, desc="Opportunities")                                                        
    for row, emb in zip(missing, embs):                                                                    
        cache[row.get('NoticeId', '')] = emb.tolist()                    
    return cache  
  
def semantic_search_rricap(opps, cache, capabilities, top_k=1):
    contract_ids = [r['NoticeId'] for r in opps if r.get('NoticeId') in cache]
    contract_embs = np.array([cache[cid] for cid in contract_ids])
    cap_embs = parallel_embed(capabilities, desc="Capabilities")
    sims = cosine_similarity(cap_embs, contract_embs)
    rricap_map = {}
    for cap_idx in range(len(capabilities)):
        for idx in sims[cap_idx].argsort()[::-1][:top_k]:
            cid = contract_ids[idx]
            rricap_map[f"CAP:{cid}"] = cache[cid]
    return rricap_map

# ---------------------------
# MAIN
# ---------------------------
def main(opps_csv, cache_json, output_json, capabilities_txt):
    print("Loading Cache")
    cache = load_cache(cache_json)
    print("Loading Boilerplate Phrases")
    boilerplate_phrases = load_boilerplate(BOILERPLATE_PATH)
    print("Filtering Data")
    opps = load_filtered_opps(opps_csv)
    print("Embedding Unembedded Contracts")
    cache = embed_missing(opps, cache, boilerplate_phrases)
    save_json(cache, cache_json, "Updated cache")
    filteredmap = {
        row['NoticeId']: cache[row['NoticeId']]
        for row in opps if row['NoticeId'] in cache
    }
    print("Loading Capabilities")
    capabilities = load_capabilities(capabilities_txt)
    # rricap_map = semantic_search_rricap(opps, cache, capabilities)
    final_output = {**filteredmap}
    save_json(final_output, output_json, "Filtered embeddings with CAP")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py opportunities.csv cache.json filtered_embeddings.json capabilities.txt")
        sys.exit(1)
    _, csvf, cachef, outf, capf = sys.argv
    main(csvf, cachef, outf, capf)
