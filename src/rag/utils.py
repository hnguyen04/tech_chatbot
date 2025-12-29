"""
Utility functions for text processing and chunking
"""
import re
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = " ".join(text.split())
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def validate_document_metadata(metadata: dict) -> bool:
    """
    Validate document metadata has required fields
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["doc_id", "source_url", "title", "category", "content_type"]
    return all(field in metadata for field in required_fields)


def safe_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Safely chunk text with error handling
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks (empty list on error)
    """
    try:
        return chunk_text(text, chunk_size, chunk_overlap)
    except Exception as e:
        print(f"Error chunking text: {e}")
        # Return original text as single chunk if chunking fails
        return [text] if text else []


def format_document_metadata(doc_id: int, source_url: str, title: str, category: str, 
                            content_type: str, chunk_index: int = 0) -> dict:
    """
    Format metadata for a document chunk
    
    Args:
        doc_id: PostgreSQL product ID
        source_url: Source URL
        title: Document title
        category: Category
        content_type: Content type (article/product)
        chunk_index: Index of chunk within document
        
    Returns:
        Metadata dictionary
    """
    return {
        "doc_id": doc_id,
        "source_url": source_url,
        "title": title,
        "category": category,
        "content_type": content_type,
        "chunk_index": chunk_index
    }

