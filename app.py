# app.py
import streamlit as st
from dotenv import load_dotenv
import os
from utils.pdf_processor import process_pdfs
from utils.vector_store import create_vectorstore
from utils.chat_handler import get_conversation_chain, handle_user_input
from templates.html_templates import css, bot_template, user_template

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = False

def main():
    # Load environment variables
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="MultiModal PDF Chat",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.header("Chat with PDFs")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        # Process button
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Process PDFs and extract content
                        chunks = process_pdfs(pdf_docs)
                        
                        # Create vectorstore
                        vectorstore = create_vectorstore(chunks)
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.processed_pdfs = True
                        
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF document.")
    
    # Main chat interface
    if st.session_state.processed_pdfs:
        # Chat input
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_user_input(user_question, st.session_state.conversation)
            
        # Display chat history (latest messages first)
        if st.session_state.chat_history:
            for i, message in enumerate(reversed(st.session_state.chat_history)):
                if (len(st.session_state.chat_history) - 1 - i) % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.info("Please upload and process your documents to start chatting.")

if __name__ == "__main__":
    main()