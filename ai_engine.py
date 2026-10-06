import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Generator, List, Union

load_dotenv()

class ChatBotEngine:
    """
    Handles the AI orchestration logic, decoupling the LLM
    implementation from the UI layer.
    """
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.llm = self._init_llm()
        self.system_message = SystemMessage(content="You are a helpful assistant.")

    def _init_llm(self):
        return init_chat_model(
            model=os.getenv("MODEL_NAME"),
            model_provider="ollama",
            temperature=self.temperature
        )

    def update_temperature(self, new_temp: float):
        """
        Updates the temperature and re-initializes the LLM.
        """
        self.temperature = new_temp
        self.llm = self._init_llm()

    def get_response_stream(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> Generator[str, None, None]:
        """
        Streams the response from the LLM.
        """
        # Ensure system message is always at the start
        full_messages = [self.system_message] + messages

        for chunk in self.llm.stream(full_messages):
            yield chunk.content

    def get_response(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> str:
        """
        Get a full non-streaming response.
        """
        full_messages = [self.system_message] + messages
        response = self.llm.invoke(full_messages)
        return response.content
