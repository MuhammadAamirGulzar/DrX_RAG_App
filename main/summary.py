import time
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import nltk
import re
import torch
from typing import Dict, Any, List
from rag import RAGQueryEngine

def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string"""
    tokens = tokenizer.encode(text)
    return len(tokens)
def generate_document_summary_from_text(full_text: str, filename: str, rag_engine: RAGQueryEngine, 
                                        max_length: int = 1000) -> Dict[str, Any]:
    """
    Generate a summary directly from full document text.
    
    Args:
        full_text: The complete text of the document
        filename: Name of the document file
        rag_engine: RAG query engine for LLM generation
        max_length: Maximum length of the summary
        
    Returns:
        Dictionary with summary and evaluation metrics
    """
    # Start timing
    start_time = time.time()
    
    # Verify we have text
    if not full_text or len(full_text.strip()) < 50:  # Arbitrary minimum length
        return {
            "summary": "Error: Document content is too short or empty to generate a meaningful summary.",
            "reference_summary": "",
            "rouge_scores": {},
            "document_length": len(full_text) if full_text else 0,
            "tokens_per_second": 0,
            "error": "Document content too short"
        }
    
    # If text is very long, select representative sections
    # This prevents context window overflow
    if len(full_text) > 10000:  # Arbitrary length limit
        # Split into paragraphs
        paragraphs = full_text.split('\n\n')
        
        # Select beginning, middle, and end paragraphs
        if len(paragraphs) > 15:
            selected_paragraphs = paragraphs[:5] + paragraphs[len(paragraphs)//2-2:len(paragraphs)//2+3] + paragraphs[-5:]
            context = '\n\n'.join(selected_paragraphs)
        else:
            context = full_text
    else:
        context = full_text
    
    # Count input tokens
    input_token_count = count_tokens(context, rag_engine.tokenizer)
    
    # Prepare the prompt for summarization
    prompt = f"""<|begin_of_text|>
<|system|>
You are a summarization assistant. Create a concise summary of the provided document.
Focus on the key points, main arguments, and important conclusions.
Be objective and comprehensive.

Document to summarize:
{context}
<|user|>
Please provide a comprehensive summary of this document. Focus on the main points and key information.
<|assistant|>"""

    # Generation process remains the same as in the original function
    inputs = rag_engine.tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        return_attention_mask=True
    )
    
    inputs = {k: v.to(rag_engine.device) for k, v in inputs.items()}
    generation_start = time.time()
    
    try:
        with torch.no_grad():
            output = rag_engine.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=rag_engine.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - generation_start
        generated_text = rag_engine.tokenizer.decode(output[0], skip_special_tokens=False)
        summary = generated_text.split("<|assistant|>")[-1].strip()
        
        if "<|" in summary:
            summary = summary.split("<|")[0].strip()
        
        output_token_count = count_tokens(summary, rag_engine.tokenizer)
        tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0
        
    except Exception as e:
        return {
            "summary": f"Error generating summary: {str(e)}",
            "reference_summary": "",
            "rouge_scores": {},
            "document_length": len(context),
            "tokens_per_second": 0,
            "error": str(e)
        }
    
    # Generate a reference summary for evaluation
    reference_summary = generate_extractive_summary(context, max_sentences=15)
    rouge_scores = evaluate_summary(summary, reference_summary)
    total_time = time.time() - start_time
    
    # Return the complete summary data
    return {
        "summary": summary,
        "reference_summary": reference_summary,
        "rouge_scores": rouge_scores,
        "document_length": len(context),
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "generation_time": generation_time,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
        "source_file": filename
    }
def generate_document_summary(chunks: List[Dict], rag_engine: RAGQueryEngine, max_length: int = 1000) -> Dict[str, Any]:
    """
    Generate a summary of the document using the chunks and RAG engine.
    
    Args:
        chunks: List of document chunks
        rag_engine: RAG query engine for LLM generation
        max_length: Maximum length of the summary
        
    Returns:
        Dictionary with summary and evaluation metrics
    """
    # Start timing
    start_time = time.time()
    
    # Verify we have chunks
    if not chunks or len(chunks) == 0:
        return {
            "summary": "Error: No document content available to summarize.",
            "reference_summary": "",
            "rouge_scores": {},
            "document_length": 0,
            "tokens_per_second": 0,
            "error": "No document content provided"
        }
    
    # If there are too many chunks, select representative ones
    if len(chunks) > 10:
        # Select beginning, middle, and end chunks for context
        selected_indices = [0, 1, len(chunks)//2, len(chunks)//2+1, len(chunks)-2, len(chunks)-1]
        selected_chunks = [chunks[i]["text"] for i in selected_indices if i < len(chunks)]
    else:
        selected_chunks = [chunk["text"] for chunk in chunks]
    
    # Join the chunks into a context for summarization
    context = "\n\n".join(selected_chunks)
    
    # Make sure we have content
    if not context or len(context.strip()) < 50:  # Arbitrary minimum length
        return {
            "summary": "Error: Document content is too short or empty to generate a meaningful summary.",
            "reference_summary": "",
            "rouge_scores": {},
            "document_length": len(context) if context else 0,
            "tokens_per_second": 0,
            "error": "Document content too short"
        }
    
    # Count input tokens
    input_token_count = count_tokens(context, rag_engine.tokenizer)
    
    # Prepare the prompt for summarization
    prompt = f"""<|begin_of_text|>
<|system|>
You are a summarization assistant. Create a concise summary of the provided document.
Focus on the key points, main arguments, and important conclusions.
Be objective and comprehensive.

Document to summarize:
{context}
<|user|>
Please provide a comprehensive summary of this document. Focus on the main points and key information.
<|assistant|>"""

    # Generate the summary using the LLM
    inputs = rag_engine.tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        return_attention_mask=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(rag_engine.device) for k, v in inputs.items()}
    
    # Start generation timing
    generation_start = time.time()
    
    # Generate
    try:
        with torch.no_grad():
            output = rag_engine.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=rag_engine.tokenizer.eos_token_id
            )
        
        # End generation timing
        generation_time = time.time() - generation_start
        
        # Decode the output
        generated_text = rag_engine.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract just the summary
        summary = generated_text.split("<|assistant|>")[-1].strip()
        
        # Remove any trailing system tokens if present
        if "<|" in summary:
            summary = summary.split("<|")[0].strip()
        
        # Count output tokens
        output_token_count = count_tokens(summary, rag_engine.tokenizer)
        
        # Calculate tokens per second
        tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0
        
    except Exception as e:
        return {
            "summary": f"Error generating summary: {str(e)}",
            "reference_summary": "",
            "rouge_scores": {},
            "document_length": len(context),
            "tokens_per_second": 0,
            "error": str(e)
        }
    
    # Generate a reference summary for evaluation
    reference_summary = generate_extractive_summary(context, max_sentences=15)
    
    # Calculate ROUGE scores
    rouge_scores = evaluate_summary(summary, reference_summary)
    
    # End timing
    total_time = time.time() - start_time
    
    return {
        "summary": summary,
        "reference_summary": reference_summary,
        "rouge_scores": rouge_scores,
        "document_length": len(context),
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "generation_time": generation_time,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second
    }

# Rest of the file remains the same
def generate_extractive_summary(text: str, max_sentences: int = 10) -> str:
    """
    Generate a simple extractive summary by selecting important sentences.
    
    Args:
        text: Input text to summarize
        max_sentences: Maximum number of sentences to include
        
    Returns:
        Extractive summary
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Simple heuristic: take the first few sentences as important context
    # and some from the middle and end
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    
    # Select sentences from beginning, middle and end
    beginning = sentences[:max_sentences//3]
    middle_start = len(sentences)//2 - max_sentences//6
    middle = sentences[middle_start:middle_start + max_sentences//3]
    end = sentences[-max_sentences//3:]
    
    # Combine selections
    selected_sentences = beginning + middle + end
    
    # Ensure we don't exceed max_sentences
    selected_sentences = selected_sentences[:max_sentences]
    
    return ' '.join(selected_sentences)

def evaluate_summary(generated_summary: str, reference_summary: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the generated summary using ROUGE metrics.
    
    Args:
        generated_summary: Generated summary text
        reference_summary: Reference summary text for comparison
        
    Returns:
        Dictionary of ROUGE scores
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference_summary, generated_summary)
    
    # Format scores for display
    result = {}
    for metric, score in scores.items():
        result[metric] = {
            'precision': round(score.precision, 4),
            'recall': round(score.recall, 4),
            'f1': round(score.fmeasure, 4)
        }
    
    return result