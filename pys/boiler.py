import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# CONFIG
# ---------------------------
INPUT_CSV = "All_Contract_Opportunities_1998_2030.csv"  # Your SAM.gov data file
OUTPUT_CSV = "boilerplate_phrases.csv"
TEXT_COLUMN = "Description"
MIN_DOC_FREQ = 0.01  # Phrase must appear in >1% of documents
NGRAM_RANGE = (2, 5)  # Look for 2- to 5-word phrases

# ---------------------------
# CLEANING FUNCTION
# ---------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"\S+@\S+", "", text)  # Remove emails
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text.strip()

# ---------------------------
# LOAD AND CLEAN DATA
# ---------------------------
try:
    # Try UTF-8 first, fallback to Latin-1 if decoding fails
    try:
        df = pd.read_csv(
            INPUT_CSV,
            encoding="utf-8",
            engine="python",         # more forgiving parser
            on_bad_lines="skip"      # skip malformed rows
        )
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed, retrying with 'latin1' encoding...")
        df = pd.read_csv(
            INPUT_CSV,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in CSV headers.")

    texts = df[TEXT_COLUMN].dropna().astype(str).apply(clean_text).tolist()
    print(f"✅ Loaded and cleaned {len(texts)} descriptions.")

except FileNotFoundError:
    print(f"❌ Could not find file: {INPUT_CSV}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit(1)

# ---------------------------
# TF-IDF PHRASE EXTRACTION
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DOC_FREQ)
X = vectorizer.fit_transform(texts)
phrases = vectorizer.get_feature_names_out()
print(f"✨ Discovered {len(phrases)} boilerplate candidates.")

# ---------------------------
# SAVE TO CSV
# ---------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Boilerplate Phrase"])
    for phrase in phrases:
        writer.writerow([phrase])

print(f"💾 Saved boilerplate phrases to {OUTPUT_CSV}")
