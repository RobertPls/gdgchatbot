from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.conf import settings
import os

# Configuración
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = os.path.join(settings.BASE_DIR, "chroma_db")

def get_vector_store():
    """Obtener o crear el vector store"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    else:
        return Chroma(embedding_function=embedding, persist_directory=PERSIST_DIR)

def split_documents(documents):
    """Dividir documentos en chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

def add_documents_to_store(documents):
    """Añadir documentos al vector store"""
    vector_store = get_vector_store()
    split_docs = split_documents(documents)
    vector_store.add_documents(split_docs)
    vector_store.persist()
    return len(split_docs)

def delete_documents_by_source(source_name: str):
    """Eliminar documentos por fuente"""
    vector_store = get_vector_store()
    
    results = vector_store.get(
        where={"source": source_name},
        include=["metadatas"]
    )
    
    if results and results.get('ids'):
        vector_store.delete(ids=results['ids'])
        return len(results['ids'])
    return 0

def replace_documents(documents, source_name: str):
    """Reemplazar documentos de una fuente específica"""
    deleted_count = delete_documents_by_source(source_name)
    
    added_count = add_documents_to_store(documents)
    
    return deleted_count, added_count