from abc import ABC, abstractmethod
from langchain_core.runnables import Runnable

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_chat_llm(self) -> Runnable:
        pass

class BaseChainFactory(ABC):
    @abstractmethod
    def create_qa_chain(self, system_prompt: str = None) -> Runnable:
        pass
    
    @abstractmethod
    def create_context_aware_chain(self, system_prompt: str = None) -> Runnable:
        pass
    
    @abstractmethod
    def create_summarization_chain(self, system_prompt: str = None) -> Runnable:
        pass