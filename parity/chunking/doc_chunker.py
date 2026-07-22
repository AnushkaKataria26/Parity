import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from markdown_it import MarkdownIt
from parity.chunking.common import EXCLUDED_DIRS

@dataclass
class DocChunk:
    file_path: str              # relative to repo root
    heading_path: str           # e.g. "Installation > Requirements"
    heading_level: int          # 1-6, or 0 for pre-heading content
    text: str                   # prose content, code fences stripped out
    code_blocks: List[str]      # fenced code block contents, stored separately
    start_line: int
    end_line: int

def discover_doc_files(repo_path: str) -> List[str]:
    discovered_files = []
    
    conventional_names = {'README', 'CHANGELOG', 'CONTRIBUTING', 'AUTHORS'}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.endswith('.egg-info')]
        
        for file in files:
            file_lower = file.lower()
            if file_lower in ('license', 'license.md', 'license.txt'):
                # Legal boilerplate is not a source of verifiable technical claims
                continue
                
            basename, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            
            is_doc = False
            if ext_lower in ('.md', '.markdown', '.rst'):
                is_doc = True
            elif ext == '' and basename.upper() in conventional_names:
                is_doc = True
                
            if is_doc:
                full_path = os.path.abspath(os.path.join(root, file))
                discovered_files.append(full_path)
                
    return sorted(discovered_files)

class _MarkdownChunkBuilder:
    def __init__(self, file_path: str, lines: List[str]):
        self.file_path = file_path
        self.lines = lines
        self.heading_path = f"{file_path} (preamble)"
        self.heading_level = 0
        self.start_line = 1
        self.code_blocks = []
        self.fence_line_ranges = [] # list of (start_idx, end_idx) 0-indexed
        
    def build(self, end_line: int) -> DocChunk:
        # Build text by including lines NOT in fence_line_ranges
        text_lines = []
        for i in range(self.start_line - 1, end_line):
            in_fence = False
            for (f_start, f_end) in self.fence_line_ranges:
                if f_start <= i < f_end:
                    in_fence = True
                    break
            if not in_fence:
                text_lines.append(self.lines[i])
                
        return DocChunk(
            file_path=self.file_path,
            heading_path=self.heading_path,
            heading_level=self.heading_level,
            text="".join(text_lines).strip(),
            code_blocks=list(self.code_blocks),
            start_line=self.start_line,
            end_line=end_line
        )

def extract_chunks_from_markdown(file_path: str, repo_root: str) -> List[DocChunk]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Warning: {file_path} has encoding issues, some characters replaced")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    rel_path = os.path.relpath(file_path, repo_root).replace(os.sep, '/')
    
    md = MarkdownIt()
    tokens = md.parse(content)
    
    chunks = []
    
    current_builder = _MarkdownChunkBuilder(rel_path, lines)
    heading_stack = [] # list of (level, text)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token.type == 'heading_open':
            # Finish the current chunk
            # The heading starts at token.map[0]. We set the end of the current chunk to that line.
            # Convert 0-indexed to 1-indexed for end_line, so token.map[0] (0-idx) is exactly the line count before the heading.
            end_line = token.map[0]
            
            # Only add preamble if it has content (or if it's the very first chunk and starts at line > 1)
            # Actually, the spec: "Content before the first heading ... (if any) -> one chunk"
            if current_builder.heading_level == 0:
                if end_line > 0:
                    chunk = current_builder.build(end_line)
                    # We might want to keep it if it has text or code blocks
                    if chunk.text or chunk.code_blocks:
                        chunks.append(chunk)
            else:
                chunks.append(current_builder.build(end_line))
            
            level = int(token.tag[1:]) # 'h1' -> 1
            
            # Extract heading text
            heading_text = ""
            i += 1
            while i < len(tokens) and tokens[i].type != 'heading_close':
                # The text inside a heading is typically an inline token
                if tokens[i].type == 'inline':
                    heading_text += tokens[i].content
                i += 1
                
            heading_text = heading_text.strip()
            
            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            
            new_heading_path = " > ".join(t for _, t in heading_stack)
            
            current_builder = _MarkdownChunkBuilder(rel_path, lines)
            current_builder.heading_path = new_heading_path
            current_builder.heading_level = level
            # Heading token's map is [start, end]
            current_builder.start_line = token.map[1] + 1
            
        elif token.type == 'fence':
            # Fenced code block
            # Add to code blocks, save line range to exclude from text
            lang = token.info.strip()
            code_content = token.content
            if lang:
                formatted_code = f"# lang: {lang}\n{code_content}"
            else:
                formatted_code = code_content
            current_builder.code_blocks.append(formatted_code)
            
            if token.map:
                current_builder.fence_line_ranges.append((token.map[0], token.map[1]))
                
        i += 1
        
    # Finish the last chunk
    end_line = len(lines)
    if current_builder.heading_level == 0:
        if end_line > 0:
            chunk = current_builder.build(end_line)
            if chunk.text or chunk.code_blocks:
                chunks.append(chunk)
    else:
        chunks.append(current_builder.build(end_line))
        
    return chunks

def extract_chunks_from_rst(file_path: str, repo_root: str) -> List[DocChunk]:
    # RST code-block directives are not stripped in this MVP; 
    # if the target repo's docs are RST-heavy, this should be revisited.
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Warning: {file_path} has encoding issues, some characters replaced")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    rel_path = os.path.relpath(file_path, repo_root).replace(os.sep, '/')
    
    chunks = []
    level_chars = []
    
    current_heading_path = f"{rel_path} (preamble)"
    current_heading_level = 0
    current_start_line = 1
    heading_stack = []
    
    # We will accumulate lines for the current chunk
    # When we find a heading, we finish the previous chunk
    
    # A heading in RST is a line of text followed by a line of punctuation
    # of at least the same length.
    punct_re = re.compile(r'^([=~\-^"\'`:.*+#])\1*\s*$')
    
    i = 0
    chunk_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Check if the next line is a valid underline
        if i + 1 < len(lines):
            next_line = lines[i+1]
            match = punct_re.match(next_line)
            if match and len(next_line.strip()) >= len(line.strip()) and line.strip():
                # Found a heading!
                char = match.group(1)
                heading_text = line.strip()
                
                # Determine level
                if char not in level_chars:
                    level_chars.append(char)
                level = level_chars.index(char) + 1
                
                # Finish current chunk (end line is i, which means it doesn't include the heading text)
                # Wait, the heading text belongs to the NEW chunk.
                # So the current chunk ends at i (1-indexed, so line number i)
                end_line = i
                
                # For preamble, only add if it has content
                text = "".join(chunk_lines).strip()
                if current_heading_level != 0 or text:
                    chunks.append(DocChunk(
                        file_path=rel_path,
                        heading_path=current_heading_path,
                        heading_level=current_heading_level,
                        text=text,
                        code_blocks=[],
                        start_line=current_start_line,
                        end_line=end_line if end_line > 0 else 1
                    ))
                    
                # Setup new chunk
                # Rebuild heading path
                # To do this correctly we need a heading stack
                pass # wait, need to maintain a stack
                
                # We'll just do a simple string append for RST since we don't have a formal stack yet.
                # Actually, let's build the stack right here.
                # We need to backtrack the stack based on level.
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))
                
                new_heading_path = " > ".join(t for _, t in heading_stack)
                
                current_heading_path = new_heading_path
                current_heading_level = level
                current_start_line = i + 2 + 1 # wait, i + 2 is the line after the underline, we want 1-indexed. Since i is 0-indexed, line after is i + 2 (0-indexed). 1-indexed that is i + 3. Let's trace: i is 0-indexed. i is heading text. i+1 is underline. i+2 is the first line of content. 1-indexed line number is (i + 2) + 1 = i + 3.
                current_start_line = i + 3
                chunk_lines = []
                i += 2
                continue
                
        chunk_lines.append(line)
        i += 1
        
    # Last chunk
    end_line = len(lines)
    text = "".join(chunk_lines).strip()
    if current_heading_level != 0 or text:
        chunks.append(DocChunk(
            file_path=rel_path,
            heading_path=current_heading_path,
            heading_level=current_heading_level,
            text=text,
            code_blocks=[],
            start_line=current_start_line,
            end_line=end_line if end_line > 0 else 1
        ))

    return chunks
