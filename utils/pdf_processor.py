# utils/pdf_processor.py
from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
import tempfile
import os

def clean_chunk_content(chunk: Any) -> str:
    """
    Clean and convert chunk content to string format.
    
    Args:
        chunk: Content chunk from unstructured
        
    Returns:
        Cleaned string content
    """
    try:
        # Try to get text representation
        content = str(chunk)
        # Remove any null bytes and strip whitespace  
        content = content.replace('\x00', '').strip()
        return content if content else "Empty content"
    except Exception as e:
        return f"Error processing content: {str(e)}"

def process_pdfs(pdf_files: List[Any]) -> List[Dict]:
    """
    Process uploaded PDF files and extract content including text, tables, and images.
    
    Args:
        pdf_files: List of uploaded PDF files from Streamlit
    
    Returns:
        List of dictionaries containing extracted content and metadata
    """
    extracted_chunks = []
    
    for pdf in pdf_files:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract content using unstructured
            chunks = partition_pdf(
                filename=tmp_file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=2000,
                combine_text_under_n_chars=1000,
                new_after_n_chars=2000
            )
            
            # Process each chunk
            for chunk in chunks:
                # Clean and prepare content
                content = clean_chunk_content(chunk)
                
                # Prepare metadata
                chunk_type = str(type(chunk).__name__)
                page_number = getattr(chunk, 'page_number', 0)
                if page_number is None:
                    page_number = 0
                
                chunk_data = {
                    'content': content,
                    'type': chunk_type,
                    'metadata': {
                        'file_name': pdf.name,
                        'page_number': page_number
                    }
                }
                
                # Handle images if present
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'image_base64'):
                    chunk_data['image'] = chunk.metadata.image_base64
                
                extracted_chunks.append(chunk_data)
        
        except Exception as e:
            # Log the error but continue processing
            print(f"Error processing PDF {pdf.name}: {str(e)}")
            continue
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
    
    return extracted_chunks