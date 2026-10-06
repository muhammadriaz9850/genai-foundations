from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Generator

class AIEngine:
    """
    Encapsulates the AI logic for the chatbot, separating
    the LLM configuration from the UI layer.
    """
    def __init__(self, model_name: str = "qwen:0.5b", temperature: float = 0.1):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.system_message = SystemMessage(content="You are a helpful assistant.")

    def get_initial_messages(self) -> List:
        """Returns the starting state of the conversation."""
        return [self.system_message]

    def stream_response(self, messages: List) -> Generator[str, None, None]:
        """
        Streams the AI response given the conversation history.
        """
        # We pass the full message list including system prompt and history
        for chunk in self.llm.stream(messages):
            yield chunk.content
