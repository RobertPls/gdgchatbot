from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import requests
import json
from datetime import datetime, timedelta, timezone
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
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'batch_size': 128,
                'normalize_embeddings': True,
            }
        )
        self.embedding_model.embed_documents(["init"])
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=200
        )
    
    def _normalize_datetime(self, date_str: str) -> datetime:
        """Normalizar cualquier formato de fecha a datetime con timezone UTC"""
        if not date_str:
            return None
        
        try:
            # Intentar diferentes formatos
            date_formats = [
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d'
            ]
            
            # Primero intentar con fromisoformat
            try:
                if date_str.endswith('Z'):
                    normalized_str = date_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(normalized_str)
                elif '+' in date_str[-6:] or '-' in date_str[-6:]:
                    dt = datetime.fromisoformat(date_str)
                else:
                    dt = datetime.fromisoformat(date_str)
                    dt = dt.replace(tzinfo=timezone.utc)
                
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                elif dt.tzinfo != timezone.utc:
                    dt = dt.astimezone(timezone.utc)
                
                return dt
                
            except ValueError:
                # Si fromisoformat falla, intentar con strptime
                for fmt in date_formats:
                    try:
                        if fmt.endswith('Z'):
                            dt = datetime.strptime(date_str, fmt)
                            dt = dt.replace(tzinfo=timezone.utc)
                        elif '%z' in fmt:
                            dt = datetime.strptime(date_str, fmt)
                            dt = dt.astimezone(timezone.utc)
                        else:
                            dt = datetime.strptime(date_str, fmt)
                            dt = dt.replace(tzinfo=timezone.utc)
                        
                        return dt
                    except ValueError:
                        continue
                
                logger.error(f"No se pudo parsear la fecha: {date_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error inesperado parseando fecha '{date_str}': {e}")
            return None

    def query(self, question: str):
        # Obtener contexto actual
        current_datetime = datetime.now(timezone.utc)
        current_date_str = current_datetime.strftime("%Y-%m-%d")
        current_time_str = current_datetime.strftime("%H:%M:%S")
        current_day_name = current_datetime.strftime("%A")
        
        # Calcular rangos temporales comunes
        next_week_start = current_datetime + timedelta(days=7)
        next_week_end = current_datetime + timedelta(days=21)
        next_month_end = current_datetime + timedelta(days=30)
        
        # Seleccionar el endpoint adecuado
        endpoint = self._select_endpoint(question)
        
        # Obtener datos en vivo desde la API
        api_data = self._get_live_api_data(endpoint)
        
        if not api_data:
            logger.error(f"No se pudieron obtener datos para el endpoint: {endpoint['name']}")
            return "Lo siento, no pude obtener datos actualizados para responder tu pregunta."
        
        # Convertir datos a documentos con contexto temporal
        documents = self._convert_to_documents(api_data, endpoint, current_datetime)
        
        # Crear vector store en memoria con los datos en vivo
        split_docs = self.text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_model
        )
        
        # Configurar retriever con búsqueda semántica
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Usar Maximal Marginal Relevance para diversidad
            search_kwargs={"k": 8, "lambda_mult": 0.5}  # Más resultados para mejor cobertura
        )
        
        # Plantilla optimizada con inteligencia temporal
        template = """Eres un experto en cultura e historia, asistente de una plataforma de eventos culturales. 
        Responde con un enfoque educativo y apasionado, combinando el contexto disponible con tu conocimiento general cuando sea apropiado.

        FECHA ACTUAL: {current_date} ({current_day}), {current_time} UTC

        CONTEXTO DISPONIBLE:
        {context}

        PREGUNTA DEL USUARIO:
        {question}

        INSTRUCCIONES PARA TU RESPUESTA:

        1. JERARQUÍA DE RESPUESTAS:
        - Si hay información relevante en el contexto: Úsala como base principal
        - Si el contexto es limitado pero la pregunta es cultural: Proporciona una respuesta educativa general
        - Si no sabes del tema: Ofrece alternativas culturales relacionadas

        2. FORMATOS DE RESPUESTA SEGÚN CASO:

        A) CUANDO HAY CONTEXTO ESPECÍFICO (eventos/artistas):
        [Nombre del Evento/Artista]
        🎨 Tipo: [Tipo de arte/evento]
        📅 Periodo: [Fecha/época histórica]
        📍 Contexto: [Detalles culturales]
        ℹ️ Valor educativo: [Explicación cultural/histórica]

        B) CUANDO ES CONOCIMIENTO CULTURAL GENERAL:
        ✨ {Tema consultado} en la cultura:
        📚 Contexto histórico: [2-3 líneas]
        🎭 Características principales: [3-5 puntos con emojis]
        🧠 Curiosidad cultural: [Dato interesante]
        🔍 Sugerencia: "Te recomendaría ver [obra/libro/museo] sobre este tema"

        3. ESTRUCTURA OBLIGATORIA:
        a) INTRODUCCIÓN:
        - Emoji + saludo cultural ("¡Qué interesante pregunta sobre arte!")
        - Validación del interés del usuario

        b) CUERPO PRINCIPAL:
        - Si hay contexto específico: Datos estructurados
        - Si es general: Explicación educativa con:
            * 1 párrafo histórico
            * 3 características clave
            * 1 curiosidad o dato sorprendente

        c) CIERRE:
        - Invitación a profundizar ("Si te interesa este tema...")
        - Emoji cultural + sugerencia (libro, museo virtual, etc.)

        4. REGLAS CLAVE:
        ✅ Prioriza el contexto cuando exista, pero no limites a solo eso
        ✅ Para preguntas culturales sin contexto: 
        - Usa tu conocimiento general educativo
        - Sé transparente: "Desde mi conocimiento cultural..."
        - Ofrece respuestas breves pero sustanciales (150-300 palabras)
        ✅ Prohibido: "No tengo información sobre eso"
        - En su lugar: "No tengo eventos registrados, pero culturalmente..."
        ✅ Usa emojis culturales (🎨🖼️📜🏛️🖌️) cada 2-3 líneas
        ✅ Mantén tono: 30% académico + 70% apasionado

        EJEMPLOS DE RESPUESTAS ACEPTABLES:
        1. Sin contexto: "¡El surrealismo de Dalí es fascinante! 🎨 Este movimiento... (explicación). Su obra más conocida... Puedes ver 'La persistencia...' en el Museo XYZ"

        2. Con contexto: "Tenemos una exposición sobre Dalí: 📅 Fechas... ℹ️ Contexto: El surrealismo..."

        RESPUESTA FINAL (usa markdown para formato):
        """
                
        prompt = ChatPromptTemplate.from_template(template)
        
        # Construir cadena de procesamiento
        chain = (
            {
                "context": retriever, 
                "question": RunnablePassthrough(), 
                "current_date": lambda _: current_date_str,
                "current_day": lambda _: current_day_name,
                "current_time": lambda _: current_time_str,
                "Tema consultado": RunnablePassthrough() 
            }
            | prompt
            | self.llm_service.factory.create_context_aware_chain()
        )
        
        # Ejecutar cadena y devolver respuesta
        result = chain.invoke(question)
        return result.content
    
    def _select_endpoint(self, question: str) -> dict:
        """Seleccionar el endpoint API basado en palabras clave en la pregunta"""
        question_lower = question.lower()
        
        if not hasattr(settings, 'JSON_API_ENDPOINTS') or not settings.JSON_API_ENDPOINTS:
            return {
                "name": "default",
                "url": "",
                "content_fields": []
            }
        
        for endpoint in settings.JSON_API_ENDPOINTS:
            if endpoint['name'].lower() in question_lower:
                return endpoint
        
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
            cache.set(cache_key, api_data, timeout=300)  
            return api_data
        
        except requests.RequestException as e:
            logger.error(f"Error de API para {endpoint['name']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            
        return None
    
    def _convert_to_documents(self, data: any, endpoint: dict, current_datetime: datetime) -> list[Document]:
        """Convertir respuesta de API a documentos de LangChain con contexto temporal"""
        documents = []
        source_name = endpoint['name']
        
        # Documento con contexto temporal global
        context_doc = f"""CONTEXTO TEMPORAL:
        - Fecha actual: {current_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")}
        - Eventos futuros: posteriores a esta fecha
        - Eventos pasados: anteriores a esta fecha
        """
        documents.append(Document(
            page_content=context_doc,
            metadata={"source": "system", "type": "temporal_context"}
        ))
        
        # Procesar eventos
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Enriquecer con contexto temporal
                    enriched_content = self._enrich_event_content(item, current_datetime)
                    
                    documents.append(Document(
                        page_content=enriched_content,
                        metadata={
                            "source": source_name,
                            "event_date": item.get('fecha', ''),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
        else:
            content = json.dumps(data, ensure_ascii=False)
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ))
        
        return documents
    
    def _enrich_event_content(self, event: dict, current_datetime: datetime) -> str:
        """Enriquecer el contenido del evento con información estructurada y contexto temporal"""
        content_parts = []
        
        # Información básica
        if event.get('nombre'):
            content_parts.append(f"EVENTO: {event['nombre']}")
        
        if event.get('artista'):
            content_parts.append(f"ARTISTA: {event['artista']}")
        
        # Manejo avanzado de fechas
        if event.get('fecha'):
            event_date = self._normalize_datetime(event['fecha'])
            if event_date:
                formatted_date = event_date.strftime("%d de %B de %Y a las %H:%M")
                content_parts.append(f"FECHA: {formatted_date}")
                
                # Días hasta el evento
                if event_date > current_datetime:
                    days_diff = (event_date - current_datetime).days
                    content_parts.append(f"DIAS_HASTA_EVENTO: {days_diff} días")
                elif event_date < current_datetime:
                    days_diff = (current_datetime - event_date).days
                    content_parts.append(f"DIAS_DESDE_EVENTO: {days_diff} días")
            else:
                content_parts.append(f"FECHA: {event['fecha']}")
        
        # Información adicional
        fields_to_include = [
            ('direccion', 'LUGAR'),
            ('categoria', 'CATEGORÍA'),
            ('genero', 'GÉNERO'),
            ('descripcionCorta', 'DESCRIPCIÓN'),
            ('descripcionGenero', 'CONTEXTO CULTURAL'),
            ('precioEntrada', 'PRECIO'),
            ('imagen', 'IMAGEN')
        ]
        
        for field, label in fields_to_include:
            if event.get(field) is not None:
                value = event[field]
                if field == 'precioEntrada':
                    value = "Gratis" if value == 0 else f"${value}"
                content_parts.append(f"{label}: {value}")
        
        return "\n".join(content_parts)