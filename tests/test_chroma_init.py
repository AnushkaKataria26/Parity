from parity.vectorstore.chroma_client import get_chroma_client, get_or_create_collections

def test_get_or_create_collections(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    client = get_chroma_client(persist_dir)
    code_col, doc_col = get_or_create_collections(client)
    
    assert code_col.name == "code_chunks"
    assert doc_col.name == "doc_chunks"

def test_get_or_create_collections_idempotent(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    client1 = get_chroma_client(persist_dir)
    get_or_create_collections(client1)
    
    client2 = get_chroma_client(persist_dir)
    code_col, doc_col = get_or_create_collections(client2)
    
    assert code_col.name == "code_chunks"
    assert doc_col.name == "doc_chunks"
