# utils/vector_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Dict, Union, Any
import uuid

def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Filter metadata to only include simple types that Chroma can handle.
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Filtered metadata dictionary with only simple types
    """
    filtered_metadata = {}
    for key, value in metadata.items():
        # Only include strings, integers, floats, and booleans
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        elif value is None:
            # Convert None to empty string
            filtered_metadata[key] = ""
    return filtered_metadata

def create_vectorstore(chunks: List[Dict]) -> Chroma:
    """
    Create a vector store from the processed PDF chunks.
    
    Args:
        chunks: List of dictionaries containing extracted content
        
    Returns:
        Chroma vector store instance
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Prepare documents for the vector store
    documents = []
    for chunk in chunks:
        # Create base metadata
        base_metadata = {
            'source': chunk['metadata'].get('file_name', ''),
            'page': chunk['metadata'].get('page_number', 0),
            'type': chunk['type'],
            'doc_id': str(uuid.uuid4())
        }
        
        # Add image data if present
        if 'image' in chunk:
            base_metadata['has_image'] = True
            # Store image data separately if needed
            # Note: We don't include the actual base64 in metadata
        else:
            base_metadata['has_image'] = False
        
        # Filter metadata to only include simple types
        filtered_metadata = filter_complex_metadata(base_metadata)
        
        # Create document
        doc = Document(
            page_content=chunk['content'],
            metadata=filtered_metadata
        )
        documents.append(doc)
    
    # Create and return vector store
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="multimodal_pdf_chat"
    )