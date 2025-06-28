from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import requests
import json
from datetime import datetime
from django.conf import settings
from langchain_core.documents import Document
from django.core.cache import cache
import logging
from apps.agent.services import LLMService

logger = logging.getLogger(__name__)

# Configuración desde settings (o valores por defecto)
EMBEDDING_MODEL = getattr(settings, 'EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = getattr(settings, 'CHUNK_SIZE', 1000)
CHUNK_OVERLAP = getattr(settings, 'CHUNK_OVERLAP', 200)

class RAGService:
    def __init__(self, provider="gemini"):
        self.llm_service = LLMService(provider)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def query(self, question: str):
        # Seleccionar el endpoint adecuado basado en la pregunta
        endpoint = self._select_endpoint(question)
        
        # Obtener datos en vivo desde la API
        api_data = self._get_live_api_data(endpoint)
        
        if not api_data:
            logger.error(f"No se pudieron obtener datos para el endpoint: {endpoint['name']}")
            return "Lo siento, no pude obtener datos actualizados para responder tu pregunta."
        
        # Convertir datos a documentos
        documents = self._convert_to_documents(api_data, endpoint)
        
        # Crear vector store en memoria con los datos en vivo
        split_docs = self.text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_model
        )
        
        # Configurar retriever con búsqueda semántica
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Plantilla para RAG
        template = """Responde la pregunta basándote únicamente en el siguiente contexto:
        {context}
        
        Pregunta: {question}
        
        Si la pregunta no puede ser respondida con el contexto, di amablemente que no tienes información actualizada.
        Respuesta:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Construir cadena de procesamiento usando el método del LLMService
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm_service.factory.create_context_aware_chain()
        )
        
        # Ejecutar cadena y devolver respuesta
        result = chain.invoke(question)
        return result.content
    
    def _select_endpoint(self, question: str) -> dict:
        """Seleccionar el endpoint API basado en palabras clave en la pregunta"""
        question_lower = question.lower()
        
        # Si no hay endpoints configurados, usar uno por defecto
        if not hasattr(settings, 'JSON_API_ENDPOINTS') or not settings.JSON_API_ENDPOINTS:
            return {
                "name": "default",
                "url": "",
                "content_fields": []
            }
        
        for endpoint in settings.JSON_API_ENDPOINTS:
            if endpoint['name'].lower() in question_lower:
                return endpoint
        
        # Si no hay coincidencia, usar el primer endpoint
        return settings.JSON_API_ENDPOINTS[0]
    
    def _get_live_api_data(self, endpoint: dict) -> any:
        """Obtener datos en vivo desde la API con caché de corta duración"""
        if not endpoint.get('url'):
            return None
            
        cache_key = f"api_data_{endpoint['name']}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            method = endpoint.get('method', 'GET').upper()
            url = endpoint['url']
            params = endpoint.get('params', {})
            headers = endpoint.get('headers', {})
            timeout = endpoint.get('timeout', 30)
            
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
                
            response.raise_for_status()
            api_data = response.json()
            
            # Almacenar en caché por 5 segundos
            cache.set(cache_key, api_data, timeout=5)
            return api_data
        
        except requests.RequestException as e:
            logger.error(f"Error de API para {endpoint['name']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            
        return None
    
    def _convert_to_documents(self, data: any, endpoint: dict) -> list[Document]:
        """Convertir respuesta de API a documentos de LangChain"""
        documents = []
        content_fields = endpoint.get('content_fields', [])
        source_name = endpoint['name']
        
        # Si se especifican campos, usarlos
        if content_fields:
            for field in content_fields:
                if field in data:
                    field_data = data[field]
                    if isinstance(field_data, list):
                        for item in field_data:
                            content = json.dumps(item, ensure_ascii=False)
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": source_name,
                                    "field": field,
                                    "timestamp": datetime.now().isoformat()
                                }
                            ))
                    else:
                        documents.append(Document(
                            page_content=f"{field}: {field_data}",
                            metadata={
                                "source": source_name,
                                "field": field,
                                "timestamp": datetime.now().isoformat()
                            }
                        ))
        else:
            # Si no hay campos específicos, usar toda la respuesta
            if isinstance(data, dict):
                content = json.dumps(data, ensure_ascii=False)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
            elif isinstance(data, list):
                for item in data:
                    content = json.dumps(item, ensure_ascii=False)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "timestamp": datetime.now().isoformat()
                        }
                    ))
        
        # Crear documento resumen
        summary = self._create_summary(data, content_fields)
        if summary:
            documents.append(Document(
                page_content=summary,
                metadata={
                    "source": source_name,
                    "type": "summary",
                    "timestamp": datetime.now().isoformat()
                }
            ))
        
        return documents
    
    def _create_summary(self, data: any, fields: list) -> str:
        """Crear resumen de los datos para contexto"""
        if not data:
            return ""
        
        if isinstance(data, dict):
            if fields:
                return "\n".join([f"{k}: {data.get(k, '')}" for k in fields if k in data])
            return json.dumps({k: v for k, v in data.items() if not isinstance(v, (dict, list))}, ensure_ascii=False)
        
        if isinstance(data, list):
            return f"Lista con {len(data)} elementos"
        
        return str(data)