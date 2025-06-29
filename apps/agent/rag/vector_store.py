# apps/agent/rag/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.conf import settings
import os
from typing import List
from langchain_core.documents import Document

# Configuración
EMBEDDING_MODEL = getattr(settings, 'EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = getattr(settings, 'CHUNK_SIZE', 1000)
CHUNK_OVERLAP = getattr(settings, 'CHUNK_OVERLAP', 200)
PERSIST_DIR = os.path.join(settings.BASE_DIR, "chroma_db")

def get_vector_store(collection_name: str = "default"):
    """Obtener o crear el vector store con una colección específica"""
    embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={
                'batch_size': 128,
                'normalize_embeddings': True
            }
        )
        
    # Fuerza la carga del modelo
    embedding.embed_query("init")    
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding,
            collection_name=collection_name
        )
    else:
        return Chroma(
            embedding_function=embedding,
            persist_directory=PERSIST_DIR,
            collection_name=collection_name
        )

def split_documents(documents: List[Document]) -> List[Document]:
    """Dividir documentos en chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

def add_documents_to_store(documents: List[Document], collection_name: str = "default") -> int:
    """Añadir documentos al vector store"""
    vector_store = get_vector_store(collection_name)
    split_docs = split_documents(documents)
    vector_store.add_documents(split_docs)
    vector_store.persist()
    return len(split_docs)

def delete_documents_by_source(source_name: str, collection_name: str = "default") -> int:
    """Eliminar documentos por fuente"""
    vector_store = get_vector_store(collection_name)
    
    results = vector_store.get(
        where={"source": source_name},
        include=["metadatas"]
    )
    
    if results and results.get('ids'):
        vector_store.delete(ids=results['ids'])
        return len(results['ids'])
    return 0

def replace_documents(documents: List[Document], source_name: str, collection_name: str = "default") -> tuple:
    """Reemplazar documentos de una fuente específica"""
    deleted_count = delete_documents_by_source(source_name, collection_name)
    added_count = add_documents_to_store(documents, collection_name)
    return (deleted_count, added_count)

def clear_collection(collection_name: str = "default") -> bool:
    """Limpiar completamente una colección"""
    try:
        vector_store = get_vector_store(collection_name)
        vector_store.delete_collection()
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False