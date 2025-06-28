from .services import LLMService
from .rag.services import RAGService
from .rag.loaders import DataLoader
from .rag.vector_store import add_documents_to_store
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
import tempfile
import os

class IngestDataView(APIView):
    parser_classes = (MultiPartParser, JSONParser)
    
    def post(self, request):
        source_type = request.data.get("source_type")  # pdf, json, sqlite
        documents = []
        
        try:
            if source_type == "pdf":
                files = request.FILES.getlist('files')
                for file in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        for chunk in file.chunks():
                            tmp.write(chunk)
                        tmp_path = tmp.name
                    
                    documents.extend(DataLoader.load_pdf(tmp_path))
                    os.unlink(tmp_path)
            
            elif source_type == "json":
                file = request.FILES['file']
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    for chunk in file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                
                content_key = request.data.get("content_key", "content")
                documents.extend(DataLoader.load_json(tmp_path, content_key))
                os.unlink(tmp_path)
            
            elif source_type == "sqlite":
                table_name = request.data.get("table_name")
                content_columns = request.data.get("content_columns", ["content"])
                metadata_columns = request.data.get("metadata_columns", [])
                
                if not table_name:
                    return Response({"error": "table_name is required"}, status=status.HTTP_400_BAD_REQUEST)
                
                documents.extend(DataLoader.load_sqlite(
                    table_name, 
                    content_columns,
                    metadata_columns
                ))
            
            else:
                return Response({"error": "Invalid source type"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Añadir documentos al vector store
            count = add_documents_to_store(documents)
            return Response({"message": f"{count} documentos añadidos"})
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatAPIView(APIView):
    def post(self, request):
        prompt = request.data.get('prompt')
        use_rag = request.data.get('use_rag', False)
        
        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        provider = request.data.get('provider', "gemini")
        
        try:
            rag_service = RAGService(provider)
            
            if use_rag:
                # Usar RAG con datos preconfigurados
                answer = rag_service.query(prompt)
                return Response({
                    "content": answer,
                    "source": "RAG",
                    "provider": provider
                })
            else:
                # Chat normal
                service = LLMService(provider)
                response = service.generate_response(prompt)
                return Response({
                    "content": response.content,
                    "source": "General Knowledge",
                    "provider": response.provider
                })
                
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        