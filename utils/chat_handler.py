# utils/chat_handler.py
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from typing import Any

def get_conversation_chain(vectorstore: Any) -> ConversationalRetrievalChain:
    """
    Create a conversation chain using the vector store.
    
    Args:
        vectorstore: Initialized vector store with documents
        
    Returns:
        ConversationalRetrievalChain instance
    """
    # Initialize language model
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-4o"  # You can change this to gpt-3.5-turbo for lower costs
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    # Create prompt template
    prompt_template = """
    You are a helpful AI assistant that can answer questions about the uploaded documents.
    Use the following context to answer the question. If you don't know the answer, just say you don't know.
    If the context includes images, use that information as well.
    
    Context: {context}
    
    Chat History: {chat_history}
    Human: {question}
    Assistant:"""
    
    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_template(prompt_template)
        }
    )
    
    return conversation_chain

def handle_user_input(user_question: str, conversation: ConversationalRetrievalChain) -> None:
    """
    Handle user input and update chat history.
    
    Args:
        user_question: User's question
        conversation: ConversationalRetrievalChain instance
    """
    if conversation is None:
        st.error("Please process documents first!")
        return
        
    response = conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']