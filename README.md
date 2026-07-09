# DT-AI-TASK-01: Industrial Offline RAG System

An offline, computationally lightweight Retrieval-Augmented Generation (RAG) search engine designed specifically for industrial technical documentation. 

This project was built to address the strict engineering constraints outlined in the task: **100% on-premises execution, limited computing resources, and aggressive hallucination control**.

## 1. Baseline Diagnosis
The provided baseline pipeline was fundamentally weak for industrial data due to three main issues:
1. **Lack of Metadata Context:** It embedded raw text without injecting the equipment ID. When a chunk just says "maximum pressure is 12 bar," the model loses the context of *which* machine it applies to.
2. **Semantic-Only Vulnerability:** Industrial docs rely on exact alphanumeric IDs (e.g., `P-200`, `E-207`). Pure dense vector search often struggles to isolate these exact matches, pulling semantically similar but incorrect machinery.
3. **Zero Hallucination Defense:** The baseline lacked an abstention mechanism. It would confidently return the closest vector match even if the user asked about a non-existent machine.

## 2. Data Quality Policy
Like real industrial documentation, the `corpus.jsonl` contained artifacts. 
* **The Issue:** System extraction tags like `[source: 70]` were polluting the raw text, which warps the mathematical placement of the document in the dense vector space.
* **The Policy (Implemented in `ingest.py`):** I utilized Regular Expressions to surgically remove these artifacts before indexing. Furthermore, I built a metadata extractor to pull hidden Equipment IDs directly from the text and titles, explicitly injecting them into a `hybrid_search_content` super-string to guarantee context preservation.

## 3. Retrieval Improvements
To solve the baseline's weaknesses, I implemented a **Hybrid Search Pipeline**:
* **Dense Semantic Search (70% Weight):** Uses the highly efficient `all-MiniLM-L6-v2` model (90MB) to understand the semantic intent of queries.
* **Sparse Lexical Search (30% Weight):** Uses `rank_bm25` (with a custom punctuation-stripping tokenizer) to lock onto exact part numbers and error codes.
* **Reciprocal Rank Fusion (RRF):** Mathematically merges both searches to ensure top documents have both semantic relevance and exact keyword matches.

## 4. Hallucination Control (Abstention)
To satisfy the strict abstention requirement, I implemented a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`).
Rather than relying on Bi-Encoder cosine distances, the Cross-Encoder evaluates the (Query + Document) pair simultaneously. If the output relevancy score falls below `0.0`, the document is aggressively filtered out. If all retrieved documents are filtered, the system successfully triggers the abstention protocol: *"I don't know the answer based on the provided documents."*

## 5. Trade-offs & Constraint Compliance
* **No Heavy Vector DBs:** To respect the "limited computing resources" and offline constraints, I avoided heavy orchestration frameworks like LangChain or local database servers (e.g., Milvus/Qdrant). The dense index runs entirely in-memory using pure `numpy` matrix multiplication, ensuring sub-millisecond execution.
* **Model Selection:** While a model like `BAAI/bge-m3` offers superior accuracy, its 2.2GB size violates the lightweight compute constraint. I deliberately chose `all-MiniLM-L6-v2` to keep the footprint minimal while relying on the Cross-Encoder for accuracy.

## 6. Evaluation & Reproducibility
Instead of using an expensive, LLM-as-a-judge framework, I built a deterministic evaluation suite (`evaluate.py`) against a custom golden dataset (`eval_queries.json`).

**Tested Scenarios:**
* **Conflict Resolution:** Successfully retrieves both documents when specs disagree (e.g., P-200 pressure at 16 bar vs 12 bar).
* **Abstention Targeting:** Successfully rejects queries about non-existent machinery.
* **Metadata Extraction:** Successfully locates hidden error codes (e.g., E-208).

### How to Run the Evaluation
Ensure you have a clean Python 3.9+ virtual environment:
```bash
pip install -r requirements.txt
python evaluate.py