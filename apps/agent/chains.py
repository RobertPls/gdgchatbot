from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from .providers import get_provider
from .interfaces import BaseChainFactory

class ChainFactory(BaseChainFactory):
    def __init__(self, provider: str = None):
        self.provider = provider
        
    def create_qa_chain(self, system_prompt: str = None) -> Runnable:
        llm = get_provider(self.provider).get_chat_llm()
        
        template = system_prompt or "Eres un asistente IA útil. Responde de forma concisa y precisa."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{input}")
        ])
        
        return prompt_template | llm
    
    def create_context_aware_chain(self, system_prompt: str = None) -> Runnable:
        llm = get_provider(self.provider).get_chat_llm()
        
        template = system_prompt or """
        Responde usando SOLAMENTE el siguiente contexto. Si no sabes la respuesta, di que no tienes información.
        
        Contexto:
        {context}
        
        Pregunta: {question}
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
    
    def create_summarization_chain(self, system_prompt: str = None) -> Runnable:
        llm = get_provider(self.provider).get_chat_llm()
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt or "Resume el siguiente texto manteniendo los puntos clave:"),
            ("human", "{text}")
        ])
        
        return prompt_template | llm

def get_chain_factory(provider: str = None) -> ChainFactory:
    return ChainFactory(provider)