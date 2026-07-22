import pytest
import math

from parity.embedding.model import embed_texts

@pytest.mark.slow
def test_real_embedding_model_output():
    # Only run this if we are intentionally running slow tests
    # which will download/load the real model.
    texts = [
        "This is a test sentence.",
        "Another sentence to check embedding dimensions.",
        ""
    ]
    
    # We do not mock anything here
    res = embed_texts(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=2)
    
    assert len(res) == 3
    
    # First sentence
    assert res[0] is not None
    assert len(res[0]) == 384
    
    # Empty string should be None
    assert res[2] is None
    
    # Check normalization: norm should be close to 1.0
    vec = res[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3
