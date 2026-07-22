"""
Embedding model and pipeline utilities for Parity.

Dependency Note:
This module relies on `sentence-transformers` for embedding code and document chunks using
the `BAAI/bge-small-en-v1.5` model. The very first time this model is loaded, `sentence-transformers`
downloads the model weights from the Hugging Face Hub (which requires internet access). 
This is a one-time setup step. After the initial download, the model is cached locally 
(e.g., in `~/.cache/huggingface`) and every subsequent call runs completely offline without any
external API calls, strictly adhering to the project's runtime constraint of no external API calls.
"""

import os
import json
import logging
from typing import List, Optional

_MODEL_CACHE = {}

def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Returns a cached SentenceTransformer instance for the given model_name.
    Loads it if not already cached.
    """
    from sentence_transformers import SentenceTransformer
    
    if model_name not in _MODEL_CACHE:
        logging.info(f"Loading embedding model '{model_name}' (this may download on first run)...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]

def embed_texts(texts: List[str], model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> List[Optional[List[float]]]:
    """
    Embeds a list of texts using the specified model in batches.
    
    Any empty or whitespace-only strings in the input are replaced with None in the output
    rather than being embedded, preserving input/output alignment. For BGE models, a specific
    retrieval prefix is added to the text prior to embedding.
    """
    if not texts:
        return []

    model = get_embedding_model(model_name)
    
    valid_indices = []
    valid_texts = []
    
    # BGE models require an instruction prefix for asymmetric retrieval tasks.
    # Omitting this measurably degrades retrieval quality.
    bge_prefix = "Represent this sentence for retrieval: " if "bge" in model_name.lower() else ""

    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(bge_prefix + text)

    results: List[Optional[List[float]]] = [None] * len(texts)
    
    if valid_texts:
        # Normalize embeddings because downstream retrieval in Phase 5 will use cosine similarity.
        # Cosine similarity is equivalent to the dot product on normalized vectors.
        # This explicitly matches Chroma's default distance metric assumption.
        embeddings = model.encode(
            valid_texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        
        for idx, emb in zip(valid_indices, embeddings):
            # emb can be a numpy array, convert to list of floats
            results[idx] = emb.tolist()
            
    return results

def extract_signature_line(source_text: str) -> str:
    """
    Extracts the signature line from a function/class definition, 
    skipping decorators.
    """
    if not source_text:
        return ""
    
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
            return line
    return ""

def build_code_chunk_embedding_text(chunk_row: dict, body_json: dict) -> str:
    """
    Builds the text to be embedded for a code chunk.
    
    The input deliberately excludes the full body text. Full bodies are noisy for the 
    retrieval task ("does this prose claim describe this code"). Instead, the embedding 
    input uses a compact representation:
    "{symbol_type} {symbol_name}\\n{docstring}\\n{signature_line}"
    This carries the actual claimable surface area.
    """
    symbol_type = chunk_row.get("symbol_type", "")
    symbol_name = chunk_row.get("symbol_name", "")
    
    docstring = body_json.get("docstring", "")
    source_text = body_json.get("source_text", "")
    
    signature_line = extract_signature_line(source_text)
    
    components = [
        f"{symbol_type} {symbol_name}".strip(),
        docstring if docstring else "",
        signature_line if signature_line else ""
    ]
    
    # Filter out completely empty components but allow normal concatenation
    text = "\\n".join(c for c in components if c).strip()
    return text

def build_doc_chunk_embedding_text(chunk_row: dict, body_json: dict) -> str:
    """
    Builds the text to be embedded for a doc chunk.
    
    Code blocks are excluded from the embedding input since they are stored separately 
    and mixing code syntax into prose degrades semantic matching.
    """
    heading = chunk_row.get("heading", "")
    text = chunk_row.get("text", "")
    
    components = [
        heading if heading else "",
        text if text else ""
    ]
    return "\\n".join(c for c in components if c).strip()

def embed_repo(conn, repo_id: int, chroma_client, code_collection, doc_collection, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> dict:
    """
    Embeds all code and doc chunks for the given repo_id and populates Chroma and SQLite.
    """
    import sqlite3
    
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # CODE CHUNKS
    cursor.execute("SELECT * FROM code_chunks WHERE repo_id = ?", (repo_id,))
    code_rows = cursor.fetchall()
    
    code_embedded = 0
    code_fallback = 0
    
    if not code_rows:
        logging.warning(f"Warning: no code chunks found for repo_id {repo_id} — did you run chunk-code first?")
    else:
        # Delete stale entries. chunk_code deletes-and-reinserts SQL rows on every run,
        # changing chunk IDs. Clean up old entries keyed to nonexistent IDs.
        code_collection.delete(where={"repo_id": repo_id})
        
        texts_to_embed = []
        for row in code_rows:
            chunk_id = row["id"]
            body_path = os.path.join("data", "code_chunk_bodies", str(repo_id), f"{chunk_id}.json")
            
            body_json = {}
            if os.path.exists(body_path):
                try:
                    with open(body_path, "r", encoding="utf-8") as f:
                        body_json = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to read {body_path}: {e}")
            else:
                logging.warning(f"Warning: missing chunk body file for chunk id {chunk_id}, using SQL fields only")
                
            text = build_code_chunk_embedding_text(dict(row), body_json)
            texts_to_embed.append(text)
            
        embeddings = embed_texts(texts_to_embed, model_name=model_name, batch_size=batch_size)
        
        # Write to Chroma and update DB
        chroma_ids = []
        chroma_embeddings = []
        chroma_metadatas = []
        
        for row, text, emb in zip(code_rows, texts_to_embed, embeddings):
            chunk_id = row["id"]
            
            if emb is None:
                logging.warning(f"Warning: chunk id {chunk_id} had empty embedding text, using symbol_name as fallback")
                code_fallback += 1
                fallback_text = row["symbol_name"]
                # Must always generate *some* embedding
                emb = embed_texts([fallback_text], model_name=model_name, batch_size=1)[0]
                
            c_id = f"code_chunk_{chunk_id}"
            chroma_ids.append(c_id)
            chroma_embeddings.append(emb)
            chroma_metadatas.append({
                "repo_id": repo_id,
                "symbol_name": row["symbol_name"],
                "file_path": row["file_path"]
            })
            
            cursor.execute("UPDATE code_chunks SET embedding_id = ? WHERE id = ?", (c_id, chunk_id))
            code_embedded += 1
            
        if chroma_ids:
            # Note: chroma allows batching too
            code_collection.add(
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                metadatas=chroma_metadatas
            )
            
        conn.commit()

    # DOC CHUNKS
    cursor.execute("SELECT * FROM doc_chunks WHERE repo_id = ?", (repo_id,))
    doc_rows = cursor.fetchall()
    
    doc_embedded = 0
    doc_fallback = 0
    
    if not doc_rows:
        logging.warning(f"Warning: no doc chunks found for repo_id {repo_id} — did you run chunk-docs first?")
    else:
        doc_collection.delete(where={"repo_id": repo_id})
        
        texts_to_embed = []
        for row in doc_rows:
            chunk_id = row["id"]
            body_path = os.path.join("data", "doc_chunk_bodies", str(repo_id), f"{chunk_id}.json")
            
            body_json = {}
            if os.path.exists(body_path):
                try:
                    with open(body_path, "r", encoding="utf-8") as f:
                        body_json = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to read {body_path}: {e}")
            else:
                logging.warning(f"Warning: missing chunk body file for chunk id {chunk_id}, using SQL fields only")
            
            text = build_doc_chunk_embedding_text(dict(row), body_json)
            texts_to_embed.append(text)
            
        embeddings = embed_texts(texts_to_embed, model_name=model_name, batch_size=batch_size)
        
        chroma_ids = []
        chroma_embeddings = []
        chroma_metadatas = []
        
        for row, text, emb in zip(doc_rows, texts_to_embed, embeddings):
            chunk_id = row["id"]
            
            if emb is None:
                logging.warning(f"Warning: doc chunk id {chunk_id} had empty embedding text, using heading as fallback")
                doc_fallback += 1
                fallback_text = row["heading"] if row["heading"] else row["file_path"]
                emb = embed_texts([fallback_text], model_name=model_name, batch_size=1)[0]
                
            c_id = f"doc_chunk_{chunk_id}"
            chroma_ids.append(c_id)
            chroma_embeddings.append(emb)
            chroma_metadatas.append({
                "repo_id": repo_id,
                "file_path": row["file_path"],
                "heading": row["heading"] or ""
            })
            
            cursor.execute("UPDATE doc_chunks SET embedding_id = ? WHERE id = ?", (c_id, chunk_id))
            doc_embedded += 1
            
        if chroma_ids:
            doc_collection.add(
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                metadatas=chroma_metadatas
            )
            
        conn.commit()

    return {
        "code_chunks_embedded": code_embedded,
        "doc_chunks_embedded": doc_embedded,
        "code_fallback_count": code_fallback,
        "doc_fallback_count": doc_fallback
    }
