from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set Ollama as default embedding model globally
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Set Ollama as default LLM globally — timeout increased to 5 minutes
Settings.llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    request_timeout=300.0
)

# Load the index you created
print("📚 Loading index...")
storage_context = StorageContext.from_defaults(persist_dir="./index")
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine()

print("\n🤖 RAG Chat Ready!")
print("Type your question. Type 'exit' to quit.\n")

# Chat loop
while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nThinking...\n")
    response = query_engine.query(question)
    print(f"AI: {response}\n")
