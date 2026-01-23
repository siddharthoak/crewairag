from langchain_ollama import ChatOllama

print("Testing ChatOllama connection...")

llm = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.7
)

try:
    response = llm.invoke("Say hello in one sentence!")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"Error: {e}")