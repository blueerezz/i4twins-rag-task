import json
import time
from retriever import HybridRetriever

def load_eval_data(filepath: str) -> list:
    """Loads the golden dataset for evaluation."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("Initializing Evaluation Pipeline...")
    # Initialize your production retriever
    retriever = HybridRetriever("./data/corpus.jsonl")
    eval_data = load_eval_data("./data/eval_queries.json")

    total_queries = len(eval_data)
    total_abstention_targets = 0
    successful_abstentions = 0
    
    exact_matches = 0
    mrr_sum = 0.0

    print("\n" + "="*60)
    print("🚀 RUNNING EVALUATION SUITE")
    print("="*60)

    for i, item in enumerate(eval_data):
        query = item["query"]
        expected = item["expected_docs"]
        category = item["category"]

        print(f"\n[Test {i+1}/{total_queries}] Category: {category.upper()}")
        print(f"Query: {query}")

        # Execute search and track latency
        start_time = time.time()
        results = retriever.search(query, top_k=3, threshold=0.0)
        latency = time.time() - start_time

        retrieved_ids = [res["doc_id"] for res in results]
        print(f"Retrieved: {retrieved_ids} | Expected: {expected} | Latency: {latency:.3f}s")

        # --- Metric 1: Abstention (Hallucination Control) ---
        if len(expected) == 0:
            total_abstention_targets += 1
            if len(retrieved_ids) == 0:
                print("✅ PASS: System correctly abstained from hallucinating.")
                successful_abstentions += 1
            else:
                print("❌ FAIL: System hallucinated an answer.")
            continue

        # --- Metric 2: Perfect Retrieval ---
        # Did we find every single document we expected? (Crucial for conflict resolution)
        missing = [doc for doc in expected if doc not in retrieved_ids]
        if not missing:
            print("✅ PASS: All required documents retrieved.")
            exact_matches += 1
        else:
            print(f"❌ FAIL: Missing documents: {missing}")

        # --- Metric 3: Mean Reciprocal Rank (MRR) ---
        # How high up the list was the FIRST correct document?
        rank = 0
        for idx, r_id in enumerate(retrieved_ids):
            if r_id in expected:
                rank = idx + 1
                break
        
        if rank > 0:
            mrr_sum += (1.0 / rank)

    # --- Print Final Report ---
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION REPORT")
    print("="*60)
    
    retrieval_queries = total_queries - total_abstention_targets
    
    print(f"Total Queries Executed:      {total_queries}")
    
    if retrieval_queries > 0:
        perfect_rate = (exact_matches / retrieval_queries) * 100
        mrr_score = mrr_sum / retrieval_queries
        print(f"Perfect Retrieval Rate:      {perfect_rate:.1f}%")
        print(f"Mean Reciprocal Rank (MRR):  {mrr_score:.3f}")
        
    if total_abstention_targets > 0:
        abstention_rate = (successful_abstentions / total_abstention_targets) * 100
        print(f"Hallucination Defense:       {abstention_rate:.1f}%")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()