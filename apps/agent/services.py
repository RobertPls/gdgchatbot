from .chains import get_chain_factory
from .schemas import LLMResponse

class LLMService:
    def __init__(self, provider: str = None):
        self.factory = get_chain_factory(provider)
        self.provider = provider or "default"
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        chain = self.factory.create_qa_chain(system_prompt)
        response = chain.invoke({"input": prompt})
        return LLMResponse(
            content=response.content,
            provider=self.provider,
            model=response.response_metadata.get('model', ''),
            metadata=response.response_metadata
        )
    
    def qa_with_context(self, question: str, context: str, system_prompt: str = None) -> LLMResponse:
        chain = self.factory.create_context_aware_chain(system_prompt)
        response = chain.invoke({"context": context, "question": question})
        return LLMResponse(
            content=response.content,
            provider=self.provider,
            model=response.response_metadata.get('model', ''),
            metadata=response.response_metadata
        )
    
    def summarize_text(self, text: str, system_prompt: str = None) -> LLMResponse:
        chain = self.factory.create_summarization_chain(system_prompt)
        response = chain.invoke({"text": text})
        return LLMResponse(
            content=response.content,
            provider=self.provider,
            model=response.response_metadata.get('model', ''),
            metadata=response.response_metadata
        )