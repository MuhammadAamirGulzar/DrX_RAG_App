import os
import torch
import tiktoken
import chromadb
import streamlit as st
import tempfile
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
import time

# Add a token counter function at the beginning of the RAGQueryEngine class


class ConversationManager:
    def __init__(self, max_history: int = 5):
        """
        Initialize a conversation manager to handle context across multiple turns.
        
        Args:
            max_history: Maximum number of QA pairs to keep in history
        """
        self.conversation_history = []
        self.max_history = max_history
        
    def add_interaction(self, question: str, answer: str, context: List[str]):
        """Add a question-answer pair to the conversation history."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "context": context
        })
        
        # Maintain only the recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self) -> str:
        """Generate a formatted conversation history string."""
        if not self.conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        for i, qa in enumerate(self.conversation_history):
            context += f"Question {i+1}: {qa['question']}\n"
            context += f"Answer {i+1}: {qa['answer']}\n\n"
        
        return context
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []



class RAGQueryEngine:
    def __init__(self, 
                 embedding_model, 
                 collection, 
                 llm_model, 
                 tokenizer, 
                 device: str = "cpu", 
                 top_k: int = 5):
        """
        Initialize the RAG Query Engine.
        
        Args:
            embedding_model: Model to generate embeddings
            collection: ChromaDB collection
            llm_model: Language model for answer generation
            tokenizer: Tokenizer for the language model
            device: Device to run models on
            top_k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.collection = collection
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self.conversation_manager = ConversationManager()
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embeddings for the query."""
        return self.embedding_model.encode([query], convert_to_tensor=False)[0]
    
    def _retrieve_relevant_chunks(self, query_embedding, top_k: int = None) -> Dict[str, Any]:
        """Retrieve the most relevant chunks from the vector database."""
        if top_k is None:
            top_k = self.top_k
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def _prepare_prompt(self, query: str, context_docs: List[str], metadata: List[Dict]) -> str:
        """Prepare the prompt for the language model with retrieved context."""
        # Get conversation history
        conversation_context = self.conversation_manager.get_conversation_context()
        
        # Join all context documents with source information
        context_text = ""
        for i, (doc, meta) in enumerate(zip(context_docs, metadata)):
            source_info = f"Source: {meta['filename']}, Page: {meta['page_number']}"
            context_text += f"Document {i+1} [{source_info}]:\n{doc}\n\n"
        
        # Format the prompt for Llama 3
        prompt = f"""<|begin_of_text|>
<|system|>
You are a helpful AI assistant. Answer the user's question based on the provided context.
If the answer is not in the context, say that you don't know or cannot find the information.
Be concise but thorough. When referring to information, cite specific document numbers when possible.

{conversation_context}

Context information:
{context_text}
<|user|>
{query}
<|assistant|>"""
        
        return prompt
    
    def _generate_answer(self, prompt: str, max_length: int = 512) -> str:
        """Generate an answer using the language model."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,  # Adjust based on model's context window
            return_attention_mask=True  # Explicitly request attention mask
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean up the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        assistant_response = generated_text.split("<|assistant|>")[-1].strip()
        
        # Remove any trailing system tokens if present
        if "<|" in assistant_response:
            assistant_response = assistant_response.split("<|")[0].strip()
            
        return assistant_response

    # Modify the query method to include token metrics
    def query(self, user_question: str) -> Tuple[str, List[str], List[Dict], Dict]:
        """
        Process a user question and generate an answer using RAG.
        
        Args:
            user_question: The user's question
            
        Returns:
            Tuple of (generated answer, retrieved documents, document metadata, performance metrics)
        """
        # Start timing
        start_time = time.time()
        
        # Generate embedding for the question
        query_embedding = self._embed_query(user_question)
        embedding_time = time.time() - start_time
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        results = self._retrieve_relevant_chunks(query_embedding)
        retrieval_time = time.time() - retrieval_start
        
        # Extract documents and metadata
        retrieved_docs = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]
        
        # Create prompt with context
        prompt_start = time.time()
        prompt = self._prepare_prompt(user_question, retrieved_docs, retrieved_metadata)
        prompt_time = time.time() - prompt_start
        
        # Count input tokens
        input_token_count = self._count_tokens(prompt)
        
        # Generate answer
        generation_start = time.time()
        answer = self._generate_answer(prompt)
        generation_time = time.time() - generation_start
        
        # Count output tokens
        output_token_count = self._count_tokens(answer)
        
        # Calculate tokens per second
        tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0
        
        # Store in conversation history
        self.conversation_manager.add_interaction(user_question, answer, retrieved_docs)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile metrics
        metrics = {
            "embedding_time": embedding_time,
            "retrieval_time": retrieval_time,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "tokens_per_second": tokens_per_second
        }
        
        return answer, retrieved_docs, retrieved_metadata, metrics
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_manager.clear_history()