import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict

# Import the pipeline , built yesterday
from ingest import process_corpus

class HybridRetriever:
    def __init__(self, corpus_path: str, model_name: str = "all-MiniLM-L6-v2", cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Hybrid Retriever and the Cross-Encoder Re-ranker.
        """
        print("Loading and cleaning corpus via ingest.py...")
        self.docs = process_corpus(corpus_path)
        
        # 1. Build the Lexical Index (BM25)
        print("Building BM25 Lexical Index...")
        tokenized_corpus = [self._tokenize(doc["hybrid_search_content"]) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Build the Dense Vector Index (NumPy)
        print(f"Loading lightweight embedding model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        print("Encoding Dense Vectors...")
        texts_to_embed = [doc["hybrid_search_content"] for doc in self.docs]
        embeddings = self.model.encode(texts_to_embed)
        
        self.vectors = np.asarray(embeddings, dtype="float32")
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        
        # 3. Load the Cross-Encoder (The Judge)
        print(f"Loading Cross-Encoder ({cross_encoder_name})...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        print("Indexing Complete.\n")

    def _tokenize(self, text: str) -> List[str]:
        """Helper method to strip punctuation and lowercase text for flawless BM25 matching."""
        clean_text = re.sub(r'[^\w\s]', '', text)
        return clean_text.lower().split()

    def search(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict]:
        """
        Executes parallel searches, merges them (RRF), and re-ranks via Cross-Encoder to filter out hallucinations.
        """
        # We fetch a slightly larger pool initially (e.g., top 5) so the Cross-Encoder has options to judge
        fetch_k = max(5, top_k * 2)
        
        # --- A. Dense Search ---
        q_vec = self.model.encode([query])[0].astype("float32")
        q_vec = q_vec / np.linalg.norm(q_vec)
        sims = self.vectors @ q_vec
        dense_top_indices = np.argsort(sims)[::-1][:fetch_k] 
        
        # --- B. Lexical Search ---
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        lexical_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # --- C. Weighted RRF ---
        rrf_scores = {}
        k_constant = 60 
        weight_dense = 0.7
        weight_lexical = 0.3
        
        for rank, doc_idx in enumerate(dense_top_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + weight_dense * (1.0 / (k_constant + rank + 1))
            
        for rank, doc_idx in enumerate(lexical_top_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + weight_lexical * (1.0 / (k_constant + rank + 1))
            
        fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # --- D. Cross-Encoder Re-ranking (The Hallucination Filter) ---
        candidates = []
        for doc_idx, rrf_score in fused_results[:fetch_k]:
            candidates.append(self.docs[doc_idx])
            
        # Prepare pairs of (Query, Document Text) for the judge
        cross_inp = [[query, doc["cleaned_text"]] for doc in candidates]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Attach the new strict scores and filter by threshold
        final_docs = []
        for i, doc in enumerate(candidates):
            doc["cross_score"] = round(float(cross_scores[i]), 4)
            # If the score is below our threshold, we actively reject it!
            if doc["cross_score"] >= threshold:
                final_docs.append(doc)
                
        # Sort by the new Cross-Encoder score and return the requested top_k
        final_docs = sorted(final_docs, key=lambda x: x["cross_score"], reverse=True)
        return final_docs[:top_k]

if __name__ == "__main__":
    retriever = HybridRetriever("./data/corpus.jsonl")
    
    test_query = "What is the maximum operating pressure for the P-200?"
    print(f"Query: '{test_query}'\n")
    print("-" * 50)
    
    # We set a threshold of 0.0. Anything below 0.0 is deemed "irrelevant" by the Cross-Encoder.
    results = retriever.search(test_query, top_k=3, threshold=0.0)
    
    if not results:
        print("ABSTENTION TRIGGERED: I don't know the answer to that question based on the provided documents.")
    else:
        for i, res in enumerate(results):
            print(f"Rank {i+1} | Relevancy Score: {res['cross_score']} | ID: {res['doc_id']}")
            print(f"Text: {res['cleaned_text']}\n")