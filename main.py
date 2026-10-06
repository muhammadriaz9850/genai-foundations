from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model="qwen:0.5b",
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