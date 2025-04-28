# app.py - Main application file
import os
from typing import List, Dict, Any, Tuple, Union, Optional
import torch
import tiktoken
import chromadb
import streamlit as st
import tempfile
import PyPDF2
import docx
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from main.vector_db import save_to_chroma
from main.text_processing import (
    read_pdf, read_docx, read_file, 
    process_file, chunk_text, get_file_type
)
from main.rag import ConversationManager, RAGQueryEngine
from main.summary import generate_document_summary, generate_extractive_summary, evaluate_summary, generate_document_summary_from_text
from main.translation import translate_document, LANGUAGE_MAP, LANGUAGE_CODE_MAP



# Initialize models and database
def init_system():
    # Create necessary directories
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("embed_model", exist_ok=True)
    os.makedirs("llm_dir", exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="document_chunks")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize embedding model
    if not os.path.exists(os.path.join("embed_model", "config.json")):
        st.info("Downloading embedding model for the first time. This may take a few minutes...")
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, 
                                  cache_folder="embed_model", device=device)
        model.save("embed_model")
    else:
        model = SentenceTransformer("embed_model", device=device, trust_remote_code=True)
    
    # Initialize LLM
    if not os.path.exists(os.path.join("llm_dir", "config.json")):
        st.info("Downloading LLM for the first time. This may take a few minutes...")
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="llm_dir")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_dir").to(device)
        tokenizer.save_pretrained("llm_dir")
        llama_model.save_pretrained("llm_dir")
    else:
        tokenizer = AutoTokenizer.from_pretrained("llm_dir")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained("llm_dir").to(device)
    
    # Create RAG engine
    rag_engine = RAGQueryEngine(
        embedding_model=model,
        collection=collection,
        llm_model=llama_model,
        tokenizer=tokenizer,
        device=device,
        top_k=5
    )
    
    return collection, model, rag_engine

# Add a helper function for cleaning up temporary files
def cleanup_temp_file(temp_file_path):
    """Safely clean up a temporary file with proper error handling"""
    try:
        # Force garbage collection first
        import gc
        gc.collect()
        
        # For Excel files, explicitly close any pandas objects
        if temp_file_path.lower().endswith(('.xlsx', '.xls')):
            for obj in gc.get_objects():
                if isinstance(obj, pd.ExcelFile) and hasattr(obj, '_path') and obj._path == temp_file_path:
                    obj.close()
        
        # Try to delete with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    st.warning(f"Could not delete temporary file: {temp_file_path}")
    except Exception as e:
        st.warning(f"Error during file cleanup: {str(e)}")


# Update the main function to include translation options
def main():
    # Disable file watcher to avoid PyTorch class registration issues
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    st.set_page_config(page_title="RAG Document Q&A System", layout="wide")
    
    # Initialize session state for conversation and summaries
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.conversation = []
        st.session_state.uploaded_files = []
        st.session_state.summaries = {}
        st.session_state.translated_files = {}  # Store information about translated files
    
    st.title("📄 Document Q&A System")
    
    # Initialize the system
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            collection, embed_model, rag_engine = init_system()
            st.session_state.collection = collection
            st.session_state.embed_model = embed_model
            st.session_state.rag_engine = rag_engine
            st.session_state.initialized = True
    else:
        collection = st.session_state.collection
        embed_model = st.session_state.embed_model
        rag_engine = st.session_state.rag_engine
    
    # Create sidebar for document upload and summarization
    with st.sidebar:
        st.header("Document Upload")
        # Updated to include Excel and CSV files
        uploaded_files = st.file_uploader("Upload PDF, DOCX, Excel or CSV files", 
                                         type=["pdf", "docx", "xlsx", "xls", "csv"], 
                                         accept_multiple_files=True)
        
        # Show chunking options only when Excel/CSV files are uploaded
        chunking_strategy = "auto"
        excel_csv_files = [f for f in uploaded_files if f.name.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if excel_csv_files:
            st.subheader("Tabular File Options")
            chunking_strategy = st.selectbox(
                "Chunking Strategy", 
                ["auto", "row", "column", "semantic"],
                help="How to chunk tabular data: by rows, columns, semantic groups, or auto-detect"
            )
        
        # Add a dictionary to track files pending translation
        if 'translation_pending' not in st.session_state:
            st.session_state.translation_pending = {}
        
        # Display uploaded files with translation and processing options
        if uploaded_files:
            st.subheader("Files Ready for Processing")
            
            for i, file in enumerate(uploaded_files):
                file_id = f"{file.name}_{i}"
                
                # Check if this file is already translated or processed
                is_processed = file.name in [f.name for f in st.session_state.uploaded_files]
                is_translated = file_id in st.session_state.translated_files
                is_pending_translation = file_id in st.session_state.translation_pending
                
                # Get file extension and determine viable options
                file_ext = os.path.splitext(file.name)[1].lower()
                
                # Display file with options
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    if is_translated:
                        # Show translated status
                        original_name = file.name
                        translated_info = st.session_state.translated_files[file_id]
                        st.write(f"📄 {original_name} → {translated_info['translated_name']}")
                    else:
                        st.write(f"📄 {file.name}")
                
                with col2:
                    # Translation button (show only if not already translated or processed)
                    if not is_translated and not is_processed and not is_pending_translation:
                        if st.button("Translate", key=f"translate_{file_id}"):
                            st.session_state.translation_pending[file_id] = {
                                "file": file,
                                "status": "pending"
                            }
                            st.rerun()
                
                with col3:
                    # Don't show process button for pending translation
                    if not is_pending_translation and not is_processed:
                        # Process button (use translated file if available)
                        process_label = "Process"
                        if is_translated:
                            process_label = "Process (Translated)"
                        
                        if st.button(process_label, key=f"process_{file_id}"):
                            # Save the file to a temporary location for processing
                            if is_translated:
                                # Use the translated file
                                translated_info = st.session_state.translated_files[file_id]
                                temp_file_path = translated_info['file_path']
                                file_name_to_use = translated_info['translated_name']
                                original_name = file.name  # Store original name for reference
                            else:
                                # Use the original file
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}")
                                temp_file.write(file.getvalue())
                                temp_file.close()
                                temp_file_path = temp_file.name
                                file_name_to_use = file.name
                                original_name = None
                            
                            with st.spinner(f"Processing {file_name_to_use}..."):
                                try:
                                    # Process the file
                                    file_type = get_file_type(file_name_to_use)
                                    
                                    # Pass chunking strategy for Excel/CSV files
                                    if file_name_to_use.lower().endswith(('.xlsx', '.xls', '.csv')):
                                        chunks = process_file(
                                            temp_file_path, 
                                            file_name_to_use, 
                                            chunking_strategy=chunking_strategy
                                        )
                                    else:
                                        chunks = process_file(temp_file_path, file_name_to_use)
                                    
                                    # Save to ChromaDB
                                    metrics = save_to_chroma(chunks, embed_model, collection)
                                    
                                    # Add metadata about translation if applicable
                                    if original_name:
                                        # Add a reference to the original file
                                        collection.update(
                                            ids=[f"file_ref_{file_name_to_use}"],
                                            metadatas=[{
                                                "original_filename": original_name,
                                                "is_translation": True,
                                                "source_language": translated_info['source_lang'],
                                                "target_language": translated_info['target_lang']
                                            }]
                                        )
                                    
                                    # Add to processed files
                                    st.session_state.uploaded_files.append(file)
                                    st.success(f"Successfully processed {file_name_to_use}")
                                    
                                    # Remove from translation pending if it was there
                                    if file_id in st.session_state.translation_pending:
                                        del st.session_state.translation_pending[file_id]
                                    
                                    # If this wasn't a translated file, clean up temp file
                                    if not is_translated:
                                        # Clean up temp file with proper error handling
                                        cleanup_temp_file(temp_file_path)
                                        
                                except Exception as e:
                                    st.error(f"Error processing file: {str(e)}")
                                    
                                    # Clean up temp file if this wasn't a translated file
                                    if not is_translated:
                                        cleanup_temp_file(temp_file_path)
            
            # Show translation dialog for pending files
            for file_id, info in list(st.session_state.translation_pending.items()):
                if info["status"] == "pending":
                    st.subheader(f"Translate '{info['file'].name}'")
                    
                    # Create form for translation options
                    source_lang = st.selectbox(
                        "Source Language", 
                        list(LANGUAGE_MAP.values()),
                        index=list(LANGUAGE_MAP.values()).index("English") if "English" in LANGUAGE_MAP.values() else 0,
                        key=f"source_lang_{file_id}"
                    )
                    
                    target_lang = st.selectbox(
                        "Target Language", 
                        list(LANGUAGE_MAP.values()),
                        index=list(LANGUAGE_MAP.values()).index("English") if "English" in LANGUAGE_MAP.values() else 0,
                        key=f"target_lang_{file_id}"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Cancel", key=f"cancel_translation_{file_id}"):
                            del st.session_state.translation_pending[file_id]
                            st.rerun()
                    
                    with col2:
                        if st.button("Start Translation", key=f"confirm_translation_{file_id}"):
                            # Get language codes from names
                            source_code = LANGUAGE_CODE_MAP[source_lang]
                            target_code = LANGUAGE_CODE_MAP[target_lang]
                            
                            # Don't translate if source and target are the same
                            if source_code == target_code:
                                st.warning("Source and target languages are the same. No translation needed.")
                                del st.session_state.translation_pending[file_id]
                                st.rerun()
                            else:
                                # Save the file to a temporary location
                                file = info["file"]
                                file_ext = os.path.splitext(file.name)[1].lower()
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                                    temp_file.write(file.getvalue())
                                    temp_file_path = temp_file.name
                                
                                # Create a path for the translated file
                                translated_file_path = os.path.join(
                                    tempfile.gettempdir(),
                                    f"translated_{os.path.basename(temp_file_path)}"
                                )
                                
                                # Create a new filename for the translated document
                                translated_name = f"{os.path.splitext(file.name)[0]}_translated_{target_code}{file_ext}"
                                
                                with st.spinner(f"Translating from {source_lang} to {target_lang}..."):
                                    success, error_msg = translate_document(
                                        temp_file_path, 
                                        translated_file_path,
                                        source_code,
                                        target_code
                                    )
                                
                                # Delete original temp file
                                cleanup_temp_file(temp_file_path)
                                
                                if success:
                                    # Store information about the translated file
                                    st.session_state.translated_files[file_id] = {
                                        "file_path": translated_file_path,
                                        "translated_name": translated_name,
                                        "source_lang": source_code,
                                        "target_lang": target_code
                                    }
                                    st.success(f"Translation complete: {file.name} → {translated_name}")
                                else:
                                    st.error(f"Translation failed: {error_msg}")
                                
                                # Remove from pending
                                del st.session_state.translation_pending[file_id]
                                st.rerun()
        
        process_btn = st.button("Process All Documents")
        
        # In the main function, within the process_btn condition:
        if process_btn and uploaded_files:
            with st.spinner("Processing documents..."):
                metrics_by_file = {}
                
                for file in uploaded_files:
                    if file.name not in [f.name for f in st.session_state.uploaded_files]:
                        # Find if this file has a translation
                        file_ids = [f"{file.name}_{i}" for i in range(len(uploaded_files))]
                        translated_file_id = next((fid for fid in file_ids if fid in st.session_state.translated_files), None)
                        
                        if translated_file_id:
                            # Process the translated version
                            translated_info = st.session_state.translated_files[translated_file_id]
                            temp_file_path = translated_info['file_path']
                            file_name_to_use = translated_info['translated_name']
                            original_name = file.name
                            is_translated = True
                        else:
                            # Process the original file
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}")
                            temp_file.write(file.getvalue())
                            temp_file.close()
                            temp_file_path = temp_file.name
                            file_name_to_use = file.name
                            original_name = None
                            is_translated = False
                        
                        try:
                            # Process the file
                            file_type = get_file_type(file_name_to_use)
                            st.info(f"Processing {file_type}: {file_name_to_use}...")
                            
                            # Pass chunking strategy for Excel/CSV files
                            if file_name_to_use.lower().endswith(('.xlsx', '.xls', '.csv')):
                                chunks = process_file(
                                    temp_file_path, 
                                    file_name_to_use, 
                                    chunking_strategy=chunking_strategy
                                )
                            else:
                                chunks = process_file(temp_file_path, file_name_to_use)
                            
                            # Save to ChromaDB with metrics
                            metrics = save_to_chroma(chunks, embed_model, collection)
                            metrics_by_file[file_name_to_use] = metrics
                            
                            # Add metadata about translation if applicable
                            if is_translated:
                                # Add a reference to the original file
                                collection.upsert(
                                    ids=[f"file_ref_{file_name_to_use}"],
                                    metadatas=[{
                                        "original_filename": original_name,
                                        "is_translation": True,
                                        "source_language": translated_info['source_lang'],
                                        "target_language": translated_info['target_lang']
                                    }]
                                )
                            
                            # Add to processed files
                            st.session_state.uploaded_files.append(file)
                            
                            # If using original file, clean up temp file
                            if not is_translated:
                                cleanup_temp_file(temp_file_path)
                                
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            
                            # If using original file, clean up temp file
                            if not is_translated:
                                cleanup_temp_file(temp_file_path)
                
                # Show metrics summary
                if metrics_by_file:
                    with st.expander("Processing Metrics"):
                        for filename, metrics in metrics_by_file.items():
                            st.subheader(f"Metrics for {filename}")
                            st.write(f"Chunks processed: {metrics['chunks_processed']}")
                            st.write(f"Tokens processed: {metrics['total_tokens']}")
                            st.write(f"Embedding time: {metrics['embedding_time']:.2f} seconds")
                            st.write(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
                
                st.success(f"Processed {len(metrics_by_file)} documents!")
            
        # Display processed files with summarization option
        if st.session_state.uploaded_files:
            st.subheader("Processed Documents")
            
            for i, file in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Check if this is a translated file
                    translated_version = False
                    for file_id, info in st.session_state.translated_files.items():
                        if file_id.startswith(file.name):
                            st.write(f"- {info['translated_name']} (Translated from {file.name})")
                            translated_version = True
                            break
                    
                    if not translated_version:
                        st.write(f"- {file.name}")
                        
                with col2:
                    # Only show summarize button for PDF and DOCX files
                    if file.name.lower().endswith(('.pdf', '.docx')):
                        if st.button("Summarize", key=f"summarize_{i}"):
                            with st.spinner(f"Summarizing {file.name}..."):
                                # First check if summary already exists in ChromaDB
                                summary_results = collection.get(
                                    where={
                                        "$and": [
                                            {"filename": {"$eq": file.name}},
                                            {"is_summary": {"$eq": True}}
                                        ]
                                    },
                                    include=["documents", "metadatas"]
                                )

                                
                                if summary_results["documents"] and len(summary_results["documents"]) > 0:
                                    # Summary exists, retrieve it
                                    summary_doc = summary_results["documents"][0]
                                    summary_data = eval(summary_doc) if isinstance(summary_doc, str) else summary_doc
                                    st.session_state.summaries[file.name] = summary_data
                                    st.success(f"Retrieved existing summary for {file.name}")
                                else:
                                    # No summary exists, read the file directly and generate summary
                                    # Re-read the original file to get full text
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}")
                                    temp_file.write(file.getvalue())
                                    temp_file.close()
                                    
                                    try:
                                        # Get full document text
                                        pages = read_file(temp_file.name)
                                        full_text = "\n\n".join([text for _, text in pages])
                                        
                                        # Generate summary from full text
                                        summary_data = generate_document_summary_from_text(full_text, file.name, rag_engine)
                                        
                                        # Check for errors
                                        if "error" in summary_data:
                                            st.error(f"Error generating summary: {summary_data.get('error', 'Unknown error')}")
                                        else:
                                            # Store summary in ChromaDB for future use
                                            summary_embedding = embed_model.encode([summary_data["summary"]], convert_to_tensor=False)[0]
                                            
                                            collection.add(
                                                ids=[f"summary_{file.name}"],
                                                embeddings=[summary_embedding],
                                                documents=[str(summary_data)],  # Store as string
                                                metadatas=[{"filename": file.name, "is_summary": True}]
                                            )
                                            
                                            st.session_state.summaries[file.name] = summary_data
                                            st.success(f"Summary generated for {file.name}")
                                    finally:
                                        # Clean up temp file with proper error handling
                                        cleanup_temp_file(temp_file.name)
                                                
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            rag_engine.clear_conversation_history()
            st.success("Conversation cleared!")
    
    # Create tabs for Q&A and Summaries only (removed Tabular Data Structure tab)
    tab1, tab2 = st.tabs(["Question & Answer", "Document Summaries"])
    
    # Rest of the code remains the same as before...
    # ...
    
    # Tab 1: Q&A Interface
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Ask Questions")
            user_question = st.text_input("Your question:", key="user_question")
            if st.button("Ask") and user_question:
                if not st.session_state.uploaded_files:
                    st.error("Please upload and process at least one document first!")
                else:
                    with st.spinner("Generating answer..."):
                        answer, docs, metadata, metrics = rag_engine.query(user_question)
                        
                        # Add to conversation history
                        st.session_state.conversation.append({
                            "question": user_question,
                            "answer": answer,
                            "sources": [
                                f"{m['filename']} ({m.get('sheet_name', 'Page')} {m.get('page_number', '')})".strip() 
                                for m in metadata
                            ],
                            "metrics": metrics
                        })
        # Show conversation history
        # st.header("Conversation")
        
        for i, exchange in enumerate(st.session_state.conversation):
            st.subheader(f"Q: {exchange['question']}")
            st.write(f"A: {exchange['answer']}")
            
            # Display sources and metrics in separate expandable sections
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Sources"):
                    for source in exchange['sources']:
                        st.write(f"• {source}")
            
            with col2:
                with st.expander("Performance Metrics"):
                    metrics = exchange.get('metrics', {})
                    st.write(f"Input tokens: {metrics.get('input_tokens', 'N/A')}")
                    st.write(f"Output tokens: {metrics.get('output_tokens', 'N/A')}")
                    st.write(f"Generation time: {metrics.get('generation_time', 0):.2f} seconds")
                    st.write(f"Tokens per second: {metrics.get('tokens_per_second', 0):.2f}")
            
            st.divider()
    
    # Tab 2: Summaries
    with tab2:
        st.header("Document Summaries")
        
        if st.session_state.summaries:
            for filename, summary_data in st.session_state.summaries.items():
                with st.expander(f"Summary for {filename}"):
                    st.subheader("Generated Summary")
                    st.write(summary_data["summary"])

                    if "rouge_scores" in summary_data and summary_data["rouge_scores"]:
                        st.subheader("ROUGE Evaluation Scores")
                        cols = st.columns(3)
                        for i, (metric, scores) in enumerate(summary_data["rouge_scores"].items()):
                            with cols[i % 3]:
                                st.metric(
                                    label=metric.upper(),
                                    value=f"F1: {scores['f1']:.4f}",
                                    delta=f"P: {scores['precision']:.4f}, R: {scores['recall']:.4f}"
                                )
                    
                    # Add performance metrics
                    st.subheader("Performance Metrics")
                    perf_cols = st.columns(3)
                    with perf_cols[0]:
                        st.metric("Input Tokens", summary_data.get("input_tokens", "N/A"))
                    with perf_cols[1]:
                        st.metric("Output Tokens", summary_data.get("output_tokens", "N/A"))
                    with perf_cols[2]:
                        st.metric("Tokens/Second", f"{summary_data.get('tokens_per_second', 0):.2f}")
                    
                    st.metric("Generation Time", f"{summary_data.get('generation_time', 0):.2f} seconds")

                    # Use a read-only text area instead of another expander
                    if "reference_summary" in summary_data:
                        st.text_area("Reference Summary (for evaluation)", summary_data["reference_summary"], height=150)

        else:
            st.info("No document summaries available. Click 'Summarize' next to a document in the sidebar to generate a summary.")

if __name__ == "__main__":
    main()





