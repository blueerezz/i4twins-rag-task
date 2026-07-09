import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict

# Import the pipeline we built yesterday
from ingest import process_corpus

class HybridRetriever:
    #  90MB model to respect compute constraints!
    def __init__(self, corpus_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Hybrid Retriever by building both Lexical and Dense indexes in memory.
        """
        print("Loading and cleaning corpus via ingest.py...")
        self.docs = process_corpus(corpus_path)
        
        # 1. Build the Lexical Index (BM25)
        print("Building BM25 Lexical Index...")
        # We use our new custom tokenizer to strip punctuation before indexing
        tokenized_corpus = [self._tokenize(doc["hybrid_search_content"]) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Build the Dense Vector Index (NumPy)
        print(f"Loading lightweight embedding model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        print("Encoding Dense Vectors...")
        texts_to_embed = [doc["hybrid_search_content"] for doc in self.docs]
        embeddings = self.model.encode(texts_to_embed)
        
        # Normalize the vectors so we can use super-fast Dot Product for Cosine Similarity
        self.vectors = np.asarray(embeddings, dtype="float32")
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        print("Indexing Complete.\n")

    def _tokenize(self, text: str) -> List[str]:
        """
        Helper method to strip punctuation and lowercase text for flawless BM25 matching.
        """
        # Remove anything that isn't a letter, number, or space, then split it.
        clean_text = re.sub(r'[^\w\s]', '', text)
        return clean_text.lower().split()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Executes parallel searches and merges them using Weighted Reciprocal Rank Fusion (RRF).
        """
        # --- A. Dense Search (Semantic meaning) ---
        q_vec = self.model.encode([query])[0].astype("float32")
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        # Fast matrix multiplication
        sims = self.vectors @ q_vec
        dense_top_indices = np.argsort(sims)[::-1][:top_k] 
        
        # --- B. Lexical Search (Exact keyword matches) ---
        # Apply the exact same punctuation stripper to the user query!
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        lexical_top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        # --- C. Weighted Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        k_constant = 60 
        
        # Custom Weights: 70% Dense, 30% Lexical
        weight_dense = 0.7
        weight_lexical = 0.3
        
        # Score Dense hits
        for rank, doc_idx in enumerate(dense_top_indices):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += weight_dense * (1.0 / (k_constant + rank + 1))
            
        # Score Lexical hits
        for rank, doc_idx in enumerate(lexical_top_indices):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += weight_lexical * (1.0 / (k_constant + rank + 1))
            
        # Sort the final fused scores
        fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Retrieve the actual document data for the final output
        final_docs = []
        for doc_idx, score in fused_results[:top_k]:
            doc_data = self.docs[doc_idx].copy()
            doc_data["rrf_score"] = round(score, 4)
            final_docs.append(doc_data)
            
        return final_docs

if __name__ == "__main__":
    # Test Execution
    retriever = HybridRetriever("./data/corpus.jsonl")
    
    # We test the conflict hidden in the documents with a punctuation mark!
    test_query = "What is the maximum operating pressure for the P-200?"
    print(f"Query: '{test_query}'\n")
    print("-" * 50)
    
    results = retriever.search(test_query)
    for i, res in enumerate(results):
        print(f"Rank {i+1} | Score: {res['rrf_score']} | ID: {res['doc_id']}")
        print(f"Text: {res['cleaned_text']}\n")