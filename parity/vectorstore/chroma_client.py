import os
import chromadb
from typing import Tuple

def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    os.makedirs(os.path.abspath(persist_dir), exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def get_or_create_collections(client: chromadb.PersistentClient) -> Tuple[chromadb.Collection, chromadb.Collection]:
    code_collection = client.get_or_create_collection(name="code_chunks", embedding_function=None)
    doc_collection = client.get_or_create_collection(name="doc_chunks", embedding_function=None)
    return code_collection, doc_collection
