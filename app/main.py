from dotenv import load_dotenv
import os

from app.vector_store import VectorStore
from app.promptbuilder import PromptBuilder
from app.llmclient import LLMClient
from app.orchestrator import RAGOrchestrator


def main():
    # 🔹 ścieżka do .env
    dotenv_path = os.path.join(os.path.dirname(__file__),"..", ".env")
    load_dotenv(dotenv_path)
    vector_store = VectorStore(dotenv_path)
    prompt_builder = PromptBuilder()
    llm_client = LLMClient(dotenv_path)

    # 🔹 orchestrator
    rag = RAGOrchestrator(vector_store, prompt_builder, llm_client)

    # 🔹 test query
    query = input("Enter your query: ")

    result = rag.run(query)

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")
    for s in result["sources"]:
        print(f"- {s['title']}")


if __name__ == "__main__":
    main()