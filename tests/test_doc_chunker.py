import os
import sqlite3
import pytest
import logging
from parity.chunking.doc_chunker import (
    discover_doc_files,
    extract_chunks_from_markdown,
    extract_chunks_from_rst,
    DocChunk
)
from parity.db.schema import SCHEMA_STATEMENTS
from parity.db.chunk_ops import store_doc_chunks

def test_extract_markdown_simple(tmp_path):
    md_content = "# Title\n\nSome paragraph."
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Title"
    assert chunks[0].heading_level == 1
    assert chunks[0].text == "Some paragraph."
    assert chunks[0].code_blocks == []

def test_extract_markdown_nested(tmp_path):
    md_content = """# H1
p1
## H2
p2
### H3
p3
## H2_back
p4"""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 4
    assert chunks[0].heading_path == "H1"
    assert chunks[1].heading_path == "H1 > H2"
    assert chunks[2].heading_path == "H1 > H2 > H3"
    assert chunks[3].heading_path == "H1 > H2_back"

def test_extract_markdown_fenced_code(tmp_path):
    md_content = """# Code Test
Here is some code:
```python
def x():
    # this is a comment
    pass
```
And some text after."""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert "def x():" not in chunks[0].text
    assert "Here is some code:" in chunks[0].text
    assert "And some text after." in chunks[0].text
    assert len(chunks[0].code_blocks) == 1
    assert "# lang: python" in chunks[0].code_blocks[0]
    assert "def x():" in chunks[0].code_blocks[0]

def test_extract_markdown_fake_heading_in_code(tmp_path):
    md_content = """# Real Heading
```
# Fake Heading
```"""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Real Heading"
    assert len(chunks[0].code_blocks) == 1
    assert "# Fake Heading" in chunks[0].code_blocks[0]

def test_extract_markdown_unclosed_fence(tmp_path):
    md_content = """# Real Heading
```
Some code that never closes..."""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    # Should not raise exception
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert len(chunks[0].code_blocks) == 1

def test_extract_markdown_empty_section(tmp_path):
    md_content = """# H1
## H2
p"""
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 2
    assert chunks[0].heading_path == "H1"
    assert chunks[0].text == ""
    assert chunks[1].heading_path == "H1 > H2"
    assert chunks[1].text == "p"

def test_extract_markdown_no_headings(tmp_path):
    md_content = "Just some text without headings."
    file_path = tmp_path / "test.md"
    file_path.write_text(md_content)
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].heading_level == 0
    assert chunks[0].heading_path.endswith(" (preamble)")
    assert chunks[0].text == "Just some text without headings."

def test_extract_markdown_empty_file(tmp_path):
    file_path = tmp_path / "test.md"
    file_path.write_text("")
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 0

def test_extract_markdown_non_utf8(tmp_path):
    file_path = tmp_path / "test.md"
    file_path.write_bytes(b"# Bad encoding\n\xff\xfe")
    
    chunks = extract_chunks_from_markdown(str(file_path), str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Bad encoding"

def test_discover_doc_files_and_exclude(tmp_path):
    (tmp_path / "README").write_text("Hello")
    (tmp_path / "docs.md").write_text("World")
    (tmp_path / "LICENSE").write_text("No")
    (tmp_path / "LICENSE.md").write_text("No")
    (tmp_path / "LICENSE.txt").write_text("No")
    
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "docs.md").write_text("No")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "docs.md").write_text("No")
    
    files = discover_doc_files(str(tmp_path))
    names = [os.path.basename(f) for f in files]
    assert "README" in names
    assert "docs.md" in names
    assert "LICENSE" not in names
    assert "LICENSE.md" not in names
    assert "LICENSE.txt" not in names
    assert len(files) == 2

def test_extract_rst(tmp_path):
    rst_content = """Title
=====

Section
-------
Text"""
    file_path = tmp_path / "test.rst"
    file_path.write_text(rst_content)
    
    chunks = extract_chunks_from_rst(str(file_path), str(tmp_path))
    assert len(chunks) == 2
    assert chunks[0].heading_path == "Title"
    assert chunks[0].heading_level == 1
    assert chunks[1].heading_path == "Title > Section"
    assert chunks[1].heading_level == 2
    assert chunks[1].text == "Text"

def test_store_doc_chunks(tmp_path, capsys):
    conn = sqlite3.connect(":memory:")
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    conn.execute("INSERT INTO repos (id, name, path) VALUES (1, 'test', 'test')")
    
    os.makedirs(tmp_path / "data" / "doc_chunk_bodies" / "1", exist_ok=True)
    
    # Needs to be run from the root of parity so it can save to 'data/doc_chunk_bodies'
    # Actually, store_doc_chunks hardcodes os.path.join("data", "doc_chunk_bodies")
    # For testing, we should mock or change cwd. Let's change cwd.
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        chunks1 = [DocChunk("f", "h1", 1, "t1", [], 1, 2)]
        store_doc_chunks(conn, 1, chunks1)
        
        cursor = conn.execute("SELECT COUNT(*) FROM doc_chunks")
        assert cursor.fetchone()[0] == 1
        
        chunks2 = [DocChunk("f", "h2", 1, "t2", [], 1, 2), DocChunk("f", "h3", 1, "t3", [], 1, 2)]
        store_doc_chunks(conn, 1, chunks2)
        
        cursor = conn.execute("SELECT COUNT(*) FROM doc_chunks")
        assert cursor.fetchone()[0] == 2
        
        # Test long heading truncation
        long_heading = "a" * 600
        chunks3 = [DocChunk("f", long_heading, 1, "t3", [], 1, 2)]
        store_doc_chunks(conn, 1, chunks3)
        
        cursor = conn.execute("SELECT heading FROM doc_chunks WHERE text = 't3'")
        stored_heading = cursor.fetchone()[0]
        assert len(stored_heading) == 500
        assert stored_heading.endswith("...")
        
        captured = capsys.readouterr()
        assert "Warning: truncating extremely long heading path" in captured.out
    finally:
        os.chdir(old_cwd)
