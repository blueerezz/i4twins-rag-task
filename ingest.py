import json
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Sanitizes raw document text by removing extraction artifacts and normalizing whitespace.
    """
    # Removes system artifacts exactly matching the pattern 
    text = re.sub(r'\\', '', text)
    
    # Normalize whitespace: replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_metadata(title: str, text: str) -> Dict[str, str]:
    """
    Extracts structured metadata (Equipment ID, Document Type) from both the title AND the body text.
    """
    metadata = {"equipment_id": "General", "doc_type": "Standard"}
    
    # 1. Extract the document classification (Everything after the em-dash)
    if " — " in title:
        parts = title.split(" — ")
        metadata["doc_type"] = parts[-1].strip()
        
    # 2. THE UPGRADE: Search both title AND text for ALL identifiers
    # re.findall pulls every match into a list
    found_ids = re.findall(r'[A-Z]+-\d+', title + " " + text)
    
    if found_ids:
        # Convert to a set to remove duplicates, then sort it cleanly
        unique_ids = sorted(list(set(found_ids)))
        # Join multiple IDs together (e.g., "E-207, E-208")
        metadata["equipment_id"] = ", ".join(unique_ids)
        
    return metadata

def process_corpus(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads the JSONL corpus, applies text cleaning, injects metadata, and 
    builds the final string optimized for both Dense and Lexical embeddings.
    """
    processed_docs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            raw_doc = json.loads(line)
            doc_title = raw_doc.get("title", "")
            raw_text = raw_doc.get("text", "")
            
            # Clean the artifacts out of the text
            cleaned_text = clean_text(raw_text)
            
            # Pull the metadata for our hybrid index
            metadata = extract_metadata(doc_title, cleaned_text)
            
            # Construct the highly-dense hybrid search string
            # This ensures the model never loses the context of what equipment it is reading about
            hybrid_search_content = (
                f"Document: {doc_title} | "
                f"Equipment: {metadata['equipment_id']} | "
                f"Type: {metadata['doc_type']} | "
                f"Context: {cleaned_text}"
            )
            
            processed_docs.append({
                "doc_id": raw_doc.get("id"),
                "title": doc_title,
                "metadata": metadata,
                "cleaned_text": cleaned_text,
                "hybrid_search_content": hybrid_search_content
            })
            
    return processed_docs

if __name__ == "__main__":
    docs = process_corpus("./data/corpus.jsonl")
    print(f"Successfully cleaned and processed {len(docs)} documents.")
    # Print DOC-07 to verify the regex cleaned the artifact and caught the ID E-207
    print(json.dumps([d for d in docs if d['doc_id'] == 'DOC-07'][0], indent=2))