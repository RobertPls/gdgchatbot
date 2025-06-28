from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from .interfaces import BaseLLMProvider

class OpenAIConfig(BaseLLMProvider):
    def get_chat_llm(self) -> ChatOpenAI:
        config = settings.LLM_CONFIG
        return ChatOpenAI(
            api_key=config["OPENAI_API_KEY"],
            model_name=config.get("OPENAI_MODEL", "gpt-4-turbo"),
            temperature=config.get("TEMPERATURE", 0.7)
        )

class GeminiConfig(BaseLLMProvider):
    def get_chat_llm(self) -> ChatGoogleGenerativeAI:
        config = settings.LLM_CONFIG
        return ChatGoogleGenerativeAI(
            model=config.get("GEMINI_MODEL", "gemini-pro"),
            google_api_key=config["GEMINI_API_KEY"],
            temperature=config.get("TEMPERATURE", 0.7),
            convert_system_message_to_human=True  
        )

class AnthropicConfig(BaseLLMProvider):
    def get_chat_llm(self) -> ChatAnthropic:
        config = settings.LLM_CONFIG
        return ChatAnthropic(
            api_key=config["ANTHROPIC_API_KEY"],
            model_name=config.get("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
            temperature=config.get("TEMPERATURE", 0.7)
        )

class OllamaConfig(BaseLLMProvider):
    def get_chat_llm(self) -> ChatOllama:
        config = settings.LLM_CONFIG
        return ChatOllama(
            base_url=config.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=config.get("OLLAMA_MODEL", "llama3"),
            temperature=config.get("TEMPERATURE", 0.7)
        )

def get_provider(provider_name: str = None) -> BaseLLMProvider:
    config = settings.LLM_CONFIG
    provider = provider_name or config["DEFAULT_PROVIDER"]
    
    providers = {
        "openai": OpenAIConfig,
        "anthropic": AnthropicConfig,
        "ollama": OllamaConfig,
        "gemini": GeminiConfig
    }
    
    if provider not in providers:
        raise ValueError(f"Proveedor no soportado: {provider}")
    
    return providers[provider]()