# app.py - Main application file
import os
from typing import List
import torch
import tiktoken
import chromadb
import streamlit as st
import tempfile
import PyPDF2
import docx
from typing import List, Dict, Any, Tuple
import time



# File handling functions
def read_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Read PDF file and extract text with page numbers.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of tuples (page_number, text)
    """
    pages = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                pages.append((i+1, text))
    return pages

def read_docx(file_path: str) -> List[Tuple[int, str]]:
    """
    Read DOCX file and extract text with page numbers.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List of tuples (page_number, text)
    """
    doc = docx.Document(file_path)
    # DOCX doesn't have native page numbers, so we'll use section breaks as a proxy
    full_text = ""
    pages = []
    page_num = 1
    
    for para in doc.paragraphs:
        if para.text.strip() == "" and full_text:  # Consider empty paragraphs as potential page breaks
            pages.append((page_num, full_text))
            full_text = ""
            page_num += 1
        else:
            full_text += para.text + "\n"
    
    # Add any remaining text
    if full_text:
        pages.append((page_num, full_text))
    
    # If no page breaks were detected, just return everything as page 1
    if not pages:
        full_text = "\n".join([para.text for para in doc.paragraphs])
        pages.append((1, full_text))
    
    return pages

def read_file(file_path: str) -> List[Tuple[int, str]]:
    """
    Read file based on extension and extract text with page numbers.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of tuples (page_number, text)
    """
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def chunk_text(text: str, filename: str, page_num: int, max_tokens: int = 512) -> List[Dict]:
    """
    Split text into token-limited chunks.
    
    Args:
        text: Text to split
        filename: Name of the source file
        page_num: Page number of the source
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({
            "chunk_number": len(chunks) + 1,
            "text": chunk_text,
            "meta_data": {"filename": filename, "page_number": page_num}
        })
    
    return chunks

def process_file(file_path: str, filename: str) -> List[Dict]:
    """
    Process a file and create chunks.
    
    Args:
        file_path: Path to the file
        filename: Name to use in metadata
        
    Returns:
        List of chunks
    """
    all_chunks = []
    pages = read_file(file_path)
    
    for page_num, text in pages:
        all_chunks.extend(chunk_text(text, filename, page_num))
    
    return all_chunks
