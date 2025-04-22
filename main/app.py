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
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
from vector_db import save_to_chroma
from text_processing import read_pdf, read_docx,read_file, process_file, chunk_text
from rag import ConversationManager, RAGQueryEngine
from summary import generate_document_summary, generate_extractive_summary, evaluate_summary, generate_document_summary_from_text


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
# Update the main function to include the summarization feature
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
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", 
                                         type=["pdf", "docx"], 
                                         accept_multiple_files=True)
        
        process_btn = st.button("Process Documents")
        
        processed_chunks = {}  # Store chunks for summarization
        # In the main function, within the process_btn condition:
        if process_btn and uploaded_files:
            with st.spinner("Processing documents..."):
                metrics_by_file = {}
                for file in uploaded_files:
                    if file.name not in [f.name for f in st.session_state.uploaded_files]:
                        # Save to temp file
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}")
                        temp_file.write(file.getvalue())
                        temp_file.close()
                        
                        # Process the file
                        st.info(f"Processing {file.name}...")
                        chunks = process_file(temp_file.name, file.name)
                        processed_chunks[file.name] = chunks  # Store chunks for summarization
                        
                        # Save to ChromaDB with metrics
                        metrics = save_to_chroma(chunks, embed_model, collection)
                        metrics_by_file[file.name] = metrics
                        
                        # Clean up
                        os.unlink(temp_file.name)
                        
                        # Add to processed files
                        st.session_state.uploaded_files.append(file)
                
                # Show metrics summary
                if metrics_by_file:
                    with st.expander("Processing Metrics"):
                        for filename, metrics in metrics_by_file.items():
                            st.subheader(f"Metrics for {filename}")
                            st.write(f"Chunks processed: {metrics['chunks_processed']}")
                            st.write(f"Tokens processed: {metrics['total_tokens']}")
                            st.write(f"Embedding time: {metrics['embedding_time']:.2f} seconds")
                            st.write(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
                
                st.success(f"Processed {len(uploaded_files)} documents!")
            

        
        # Display processed files with summarization option
        if st.session_state.uploaded_files:
            st.subheader("Processed Documents")
            
            for i, file in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"- {file.name}")
                with col2:
                    # In the main function, within the summarize button condition:
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
                                
                                # Get full document text
                                pages = read_file(temp_file.name)
                                full_text = "\n\n".join([text for _, text in pages])
                                
                                # Clean up temp file
                                os.unlink(temp_file.name)
                                
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
                                                
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            rag_engine.clear_conversation_history()
            st.success("Conversation cleared!")
    
    # Create tabs for Q&A and Summaries
    tab1, tab2 = st.tabs(["Question & Answer", "Document Summaries"])
    
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
                            "sources": [f"{m['filename']} (Page {m['page_number']})" for m in metadata],
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
