# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system for document question answering and summarization built with Streamlit, ChromaDB, and Meta's Llama model.

## 🌟 Features

- **Document Question Answering**: Ask questions about your documents and get accurate answers with source attribution
- **Document Summarization**: Generate comprehensive summaries of uploaded documents
- **Multiple File Format Support**: Process PDF, DOCX, Excel (XLSX/XLS), and CSV files
- **Tabular Data Processing**: Intelligent chunking of Excel and CSV data with multiple strategies
- **Document Translation**: Translate documents from multiple languages to English before processing
- **Conversation History**: Maintain context across multiple questions
- **Performance Metrics**: Track token usage and generation time

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- PyTorch
- RAM (8GB+ recommended)
- GPU acceleration recommended but can be run on CPU

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadAamirGulzar/DrX_RAG_App.git
   cd DrX-RAG-APP
   ```

2. Create and activate a virtual environment using environment file:
   ```bash
    conda env create -f environment.yml
   ```

3. Install dependencies (optional if you dont create venv using environment.yml file):
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (required for summarization):
   ```python
   import nltk
   nltk.download('punkt')
   ```

## 🚀 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8502)

3. Upload documents:
   - Click the "Upload PDF, DOCX, Excel or CSV files" button in the sidebar
   - Select one or more files from your computer
   - For Excel/CSV files, select a chunking strategy (auto, row, column, or semantic)
   - Click "Process Documents" to extract, embed text and store in chromaDB

4. Translate documents (optional):
   - Click "Translate" button next to an uploaded document
   - Select source language and target language
   - Click "Start Translation" to translate the document before processing
   - After translation completes, click "Process" to embed and store the translated content

5. Ask questions:
   - Type your question in the input field
   - Click "Ask" to generate an answer
   - View the sources used and performance metrics

6. Generate summaries:
   - Click "Summarize" next to a document in the sidebar
   - The summary will appear in the "Document Summaries" tab
   - Summaries include ROUGE evaluation metrics and performance statistics

## 🧩 Project Structure

- `app.py`: Main application file with Streamlit UI
- `text_processing.py`: Text reading and chunking
- `excel_processing.py`: Excel and CSV file processing
- `vector_db.py`: Vector db data collection locally
- `rag.py`: Core RAG functionality and conversation management
- `summary.py`: Document summarization capabilities
- `translation.py`: Document translation functionality

## 🔧 Configuration

The system uses these default settings, which can be modified in the code:

- Maximum conversation history: 5 QA pairs
- Top-k retrieval: 5 document chunks per query
- Maximum chunk size: 512 tokens
- Maximum summary length: 1000 tokens
- Default target language: English
- Excel/CSV chunking strategy: Auto-detect

## 🤔 How It Works

1. **Document Processing**:
   - Documents are uploaded and optionally translated
   - Text is extracted and split into chunks
   - For tabular data, intelligent chunking strategies are applied based on data structure
   - Chunks are embedded using SentenceTransformer and stored in ChromaDB

2. **Translation**:
   - Documents can be translated before processing
   - Multiple language pairs supported (e.g., French→English, Urdu→English)
   - Document structure is preserved during translation
   - Translated documents are processed and embedded in English for better semantic matching

3. **Question Answering**:
   - User question is embedded
   - Most semantically similar document chunks are retrieved
   - A prompt is constructed with the question and retrieved context
   - LLama 3.2 1B Instruct model generates an answer

4. **Summarization**:
   - Full document text is processed directly
   - For large documents, representative sections are selected
   - An abstractive summary is generated using the language model
   - An extractive summary is created for ROUGE evaluation
   - The summary is stored in ChromaDB for future retrieval

## 🔄 Memory Optimization

The system is optimized for memory efficiency:
- Document text is loaded only when needed
- Text is cleared from memory after processing
- Only summaries are stored persistently
- Automatic device detection for GPU acceleration
- Temporary files cleaned up properly after processing

## ⚠️ Limitations

- Context window limited by the model's maximum token count
- Very large documents may need representative selection
- Quality of answers depends on the quality of retrieved chunks
- Translation quality depends on the language model and complexity of content
- Performance may vary based on hardware capabilities

## 📚 Required Libraries

- `streamlit`: For the web interface
- `torch`: For PyTorch operations
- `transformers`: For the LLM backend and translation models
- `sentence-transformers`: For embedding generation
- `chromadb`: For vector database storage
- `PyPDF2`: For PDF processing
- `python-docx`: For DOCX processing
- `pandas`: For Excel/CSV processing
- `openpyxl`: For Excel file handling
- `fpdf`: For PDF generation after translation
- `nltk`: For text processing and evaluation
- `rouge-score`: For summary evaluation
- `sentencepiece`: For translation model tokenization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.