# Multi-Format RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) application for document intelligence across PDF, DOCX, XLSX/XLS, and CSV.

The system combines semantic retrieval, multilingual translation, conversational Q&A, and document summarization in a single Streamlit interface.

## What This Product Solves

Teams often need answers from mixed-format documents without building separate pipelines for narrative text and tabular data. This application provides:

- context-grounded answers with source traceability
- robust handling for text and spreadsheet data
- optional translation before indexing for multilingual workflows
- summary generation with lightweight quality scoring

## Core Capabilities

- Multi-format ingestion: PDF, DOCX, XLSX/XLS, CSV
- Retrieval-augmented question answering with conversation history
- Tabular chunking strategies: row-based, column-based, semantic, auto-inferred
- Document translation pipeline with format-preserving output
- Document summarization with ROUGE-based reference scoring
- Runtime metrics for embedding and generation performance

## System Architecture

Detailed design notes are documented in `ARCHITECTURE.md`. High-level flow:

1. Ingest and parse files by type.
2. Optionally translate source content.
3. Chunk text/tabular content and generate embeddings.
4. Persist chunks in ChromaDB.
5. Retrieve top-k chunks for user queries.
6. Build prompt with conversation memory and source context.
7. Generate grounded responses with Llama 3.2 1B Instruct.

## Tech Stack

- Application layer: Streamlit
- LLM inference: Hugging Face Transformers + PyTorch
- Retriever embeddings: sentence-transformers (nomic-ai/nomic-embed-text-v2-moe)
- Vector store: ChromaDB (persistent local collection)
- Document processing: PyPDF2, python-docx, pandas, openpyxl
- Translation: Helsinki-NLP Marian models with multilingual fallback
- Evaluation: NLTK sentence tokenization + ROUGE

## Repository Layout

- `app.py` - Streamlit UI and orchestration workflow
- `main/rag.py` - retrieval and answer generation engine
- `main/text_processing.py` - file parsing and chunk preparation
- `main/excel_processing.py` - tabular structure analysis and chunking strategies
- `main/translation.py` - translation workflow per document type
- `main/summary.py` - abstractive summary generation and ROUGE scoring
- `main/vector_db.py` - embedding generation and Chroma persistence

## Local Setup

### Prerequisites

- Python 3.11+ recommended
- 8 GB RAM minimum, 16 GB preferred
- CUDA-capable GPU optional (CPU mode supported)

### Install

```bash
git clone https://github.com/MuhammadAamirGulzar/DrX_RAG_App.git
cd DrX_RAG_App
```

Option 1: Conda

```bash
conda env create -f environment.yml
conda activate drx-rag-app
```

Option 2: venv + pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download required NLTK tokenizer resource:

```python
import nltk
nltk.download("punkt")
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload documents, process them, then use the Q&A and Summaries tabs.

## Operational Notes

- First run downloads and caches embedding + generation models.
- ChromaDB persistence path: `chroma_db/`.
- Local model cache directories: `embed_model/` and `llm_dir/`.
- Large documents are summarized using representative-section selection to reduce context overflow.

## Performance Considerations

- Retrieval quality is sensitive to chunking strategy and document cleanliness.
- Generation latency depends on model size, token count, and hardware.
- Translation adds pre-indexing latency but improves cross-language retrieval consistency.

## Known Limitations

- Long-context behavior is constrained by model token limits.
- Summary quality varies by source structure and language complexity.
- Table-heavy files with ambiguous headers may require explicit chunking strategy selection.

## Suggested GitHub Metadata

- Repository name: `multiformat-rag-assistant`
- Description: `Multi-format RAG app with translation, semantic retrieval, and summarization for PDF, DOCX, Excel, and CSV.`
- Topics: `rag`, `retrieval-augmented-generation`, `streamlit`, `llm`, `chromadb`, `sentence-transformers`, `transformers`, `document-qa`, `summarization`, `machine-translation`, `python`, `vector-search`

## License

Licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See `LICENSE` for details.
