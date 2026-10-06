import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    model_provider="ollama",
    temperature=0.1
)
messages = [
    SystemMessage(content="You are a helpful assistant.")
]
while True:
    question = input("You: ").strip()

    if not question:
       continue
    if question in ("exit", "quit", "bye"):
       print("Goodbye!")
       break

    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print(f"Bot: {response.content}")