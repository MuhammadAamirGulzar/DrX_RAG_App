# app.py - Main application file
import os
from typing import List
import torch
import tiktoken
import chromadb
import streamlit as st
from typing import List, Dict, Any, Tuple
import time



def save_to_chroma(chunks: List[Dict], embed_model, collection, is_summary=False):
    """
    Generate embeddings and save chunks to ChromaDB.
    
    Args:
        chunks: List of text chunks
        embed_model: Embedding model
        collection: ChromaDB collection
        is_summary: Whether the chunks are summaries
    Returns:
        Dictionary with performance metrics
    """
    if not chunks:
        return {"tokens_per_second": 0, "embedding_time": 0, "chunks_processed": 0}
    
    start_time = time.time()
    
    texts = [chunk["text"] for chunk in chunks]
    total_tokens = sum(len(text.split()) for text in texts)  # Rough token estimate
    
    # Generate embeddings
    embedding_start = time.time()
    embeddings = embed_model.encode(texts, convert_to_tensor=False)
    embedding_time = time.time() - embedding_start
    
    ids = [f"chunk_{i}_{chunks[i]['meta_data']['filename']}_{chunks[i]['meta_data']['page_number']}" 
           for i in range(len(chunks))]
    documents = texts
    metadatas = [chunk["meta_data"] for chunk in chunks]
    
    # Add is_summary flag to metadata if needed
    if is_summary:
        for meta in metadatas:
            meta["is_summary"] = True

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    total_time = time.time() - start_time
    tokens_per_second = total_tokens / embedding_time if embedding_time > 0 else 0
    
    return {
        "tokens_per_second": tokens_per_second,
        "embedding_time": embedding_time,
        "total_time": total_time,
        "chunks_processed": len(chunks),
        "total_tokens": total_tokens
    }