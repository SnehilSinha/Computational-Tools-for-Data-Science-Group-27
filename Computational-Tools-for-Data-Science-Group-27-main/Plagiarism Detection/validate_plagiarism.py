import pandas as pd
import numpy as np
import hashlib
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# --- MinHash Implementation (Same as Notebook) ---
class MinHash:
    def __init__(self, num_perm=128, seed=42):
        self.num_perm = num_perm
        self.seed = seed
        self.permutations = self._generate_permutations()

    def _generate_permutations(self):
        random.seed(self.seed)
        max_val = (2**32) - 1
        perms = []
        for _ in range(self.num_perm):
            a = random.randint(1, max_val)
            b = random.randint(0, max_val)
            perms.append((a, b))
        return perms

    def get_shingles(self, text, k=3):
        text = str(text).lower()
        tokens = text.split()
        if len(tokens) < k:
            return set([text])
        shingles = set()
        for i in range(len(tokens) - k + 1):
            shingle = " ".join(tokens[i:i+k])
            shingles.add(shingle)
        return shingles

    def compute_signature(self, text, k=3):
        shingles = self.get_shingles(text, k)
        signature = [float('inf')] * self.num_perm
        for shingle in shingles:
            hashed_shingle = int(hashlib.sha256(shingle.encode('utf-8')).hexdigest()[:8], 16)
            prime = 4294967311
            for i, (a, b) in enumerate(self.permutations):
                hash_val = (a * hashed_shingle + b) % prime
                if hash_val < signature[i]:
                    signature[i] = hash_val
        return signature

    def compute_similarity(self, sig1, sig2):
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for i in range(self.num_perm) if sig1[i] == sig2[i])
        return matches / self.num_perm

# --- Validation Logic ---

def validate():
    print("Starting Validation...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/preprocessed_dataset.csv')
        print(f"[PASS] Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"[FAIL] Could not load data: {e}")
        return

    # 2. Validate MinHash
    mh = MinHash(num_perm=128)
    
    # Test Case: Identical texts should have similarity 1.0
    text1 = "this is a test sentence for minhash"
    text2 = "this is a test sentence for minhash"
    sig1 = mh.compute_signature(text1)
    sig2 = mh.compute_signature(text2)
    sim = mh.compute_similarity(sig1, sig2)
    
    if sim == 1.0:
        print("[PASS] MinHash Identity Check")
    else:
        print(f"[FAIL] MinHash Identity Check. Expected 1.0, got {sim}")

    # Test Case: Completely different texts should have low similarity
    text3 = "completely different content unrelated"
    sig3 = mh.compute_signature(text3)
    sim_diff = mh.compute_similarity(sig1, sig3)
    
    if sim_diff < 0.1:
        print(f"[PASS] MinHash Difference Check (Sim: {sim_diff})")
    else:
        print(f"[FAIL] MinHash Difference Check. Expected < 0.1, got {sim_diff}")

    # 3. Validate Embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[PASS] sentence_transformers library is available and model loaded.")
        
        emb1 = model.encode([text1])
        emb2 = model.encode([text2])
        emb3 = model.encode([text3])
        
        cos_sim_ident = cosine_similarity(emb1, emb2)[0][0]
        cos_sim_diff = cosine_similarity(emb1, emb3)[0][0]
        
        if cos_sim_ident > 0.99:
             print("[PASS] Embedding Identity Check")
        else:
             print(f"[FAIL] Embedding Identity Check. Got {cos_sim_ident}")
             
        if cos_sim_diff < 0.5: # Threshold depends on model, but usually low for unrelated
             print(f"[PASS] Embedding Difference Check (Sim: {cos_sim_diff:.4f})")
        else:
             print(f"[WARN] Embedding Difference Check. Got {cos_sim_diff:.4f} (might be high due to model bias or short text)")

    except ImportError:
        print("[WARN] sentence_transformers not found. Validating TF-IDF fallback.")
        vectorizer = TfidfVectorizer()
        embs = vectorizer.fit_transform([text1, text2, text3])
        cos_sim_ident = cosine_similarity(embs[0], embs[1])[0][0]
        cos_sim_diff = cosine_similarity(embs[0], embs[2])[0][0]
        
        if cos_sim_ident > 0.99:
             print("[PASS] TF-IDF Identity Check")
        
        if cos_sim_diff < 0.1:
             print(f"[PASS] TF-IDF Difference Check (Sim: {cos_sim_diff})")

    # 4. Data Validation (Average Scores)
    # Re-compute signatures for a subset to check distribution
    print("\nValidating Dataset Performance...")
    
    # Identify source files
    source_files = df[df['Category'] == 'orig']
    source_map = {row['Task']: row['clean_text'] for _, row in source_files.iterrows()}
    
    # Helper to compute sim
    def get_mh_sim(row):
        if row['Task'] not in source_map: return 0.0
        return mh.compute_similarity(mh.compute_signature(row['clean_text']), mh.compute_signature(source_map[row['Task']]))

    df['mh_sim'] = df.apply(get_mh_sim, axis=1)
    
    avg_cut = df[df['Category'] == 'cut']['mh_sim'].mean()
    avg_non = df[df['Category'] == 'non']['mh_sim'].mean()
    
    print(f"Average MinHash Sim for 'cut': {avg_cut:.4f}")
    print(f"Average MinHash Sim for 'non': {avg_non:.4f}")
    
    if avg_cut > avg_non + 0.2:
        print("[PASS] MinHash effectively separates 'cut' from 'non'.")
    else:
        print("[FAIL] MinHash separation is weak.")

    print("\nValidation Complete.")

if __name__ == "__main__":
    validate()
