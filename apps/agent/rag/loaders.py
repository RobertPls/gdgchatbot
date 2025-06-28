# data_loader.py - Actualizado para trabajar con APIs JSON
import json
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from datetime import datetime
from typing import List, Dict, Any, Union
import os

class DataLoader:
    @staticmethod
    def load_pdf(file_path: str):
        """Cargar datos desde un archivo PDF"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF no encontrado: {file_path}")
        
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_json(file_path: str, content_key: str = "content"):
        """Cargar datos desde un archivo JSON local"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON no encontrado: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return DataLoader._convert_json_to_documents(data, file_path)
    
    @staticmethod
    def load_from_api(url: str, method: str = "GET", params: Dict = None, 
                     headers: Dict = None, timeout: int = 30) -> List[Document]:
        """Cargar datos directamente desde una API JSON"""
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            return DataLoader._convert_json_to_documents(data, url)
            
        except requests.RequestException as e:
            raise Exception(f"Error cargando datos de API: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error decodificando JSON: {str(e)}")
    
    @staticmethod
    def _convert_json_to_documents(data: Any, source: str) -> List[Document]:
        """Convertir cualquier estructura JSON a documentos"""
        documents = []
        
        def process_data(obj: Any, path: str = "", parent_key: str = "") -> List[Document]:
            docs = []
            
            if isinstance(obj, dict):
                # Crear documento principal para el objeto
                content = DataLoader._dict_to_readable_text(obj)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "data_type": "json_object",
                        "path": path or "root",
                        "parent_key": parent_key,
                        "load_time": datetime.now().isoformat()
                    }
                )
                docs.append(doc)
                
                # Procesar elementos anidados
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, (dict, list)) and value:
                        nested_docs = process_data(value, current_path, key)
                        docs.extend(nested_docs)
                    elif value is not None:
                        # Crear documento para campos importantes
                        field_doc = Document(
                            page_content=f"{key}: {value}",
                            metadata={
                                "source": source,
                                "data_type": "json_field",
                                "field_name": key,
                                "field_value": str(value),
                                "path": current_path,
                                "load_time": datetime.now().isoformat()
                            }
                        )
                        docs.append(field_doc)
                        
            elif isinstance(obj, list):
                # Procesar lista
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]" if path else f"item_{i}"
                    nested_docs = process_data(item, current_path, f"{parent_key}_item_{i}")
                    docs.extend(nested_docs)
            
            return docs
        
        return process_data(data, source=source)
    
    @staticmethod
    def _dict_to_readable_text(data: Dict[str, Any], indent: int = 0) -> str:
        """Convertir diccionario a texto legible"""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(DataLoader._dict_to_readable_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  Item {i + 1}:")
                        lines.append(DataLoader._dict_to_readable_text(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def load_json_with_fields(data: Dict[str, Any], content_fields: List[str], source: str) -> List[Document]:
        """Cargar solo campos específicos del JSON"""
        documents = []
        
        # Procesar campos específicos
        for field in content_fields:
            if field in data:
                field_data = data[field]
                field_docs = DataLoader._convert_json_to_documents(field_data, f"{source}_{field}")
                documents.extend(field_docs)
        
        # Crear documento resumen
        summary_content = DataLoader._create_summary_from_fields(data, content_fields)
        if summary_content:
            summary_doc = Document(
                page_content=summary_content,
                metadata={
                    "source": source,
                    "data_type": "summary",
                    "fields_included": ",".join(content_fields),
                    "load_time": datetime.now().isoformat()
                }
            )
            documents.append(summary_doc)
        
        return documents
    
    @staticmethod
    def _create_summary_from_fields(data: Dict[str, Any], fields: List[str]) -> str:
        """Crear resumen a partir de campos específicos"""
        summary_parts = []
        
        for field in fields:
            if field in data:
                value = data[field]
                if isinstance(value, (dict, list)):
                    summary_parts.append(f"{field}: {json.dumps(value, ensure_ascii=False, indent=2)}")
                else:
                    summary_parts.append(f"{field}: {value}")
        
        if summary_parts:
            return f"Resumen de datos cargados:\n" + "\n".join(summary_parts)
        
        return ""