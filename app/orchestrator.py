from app.embeddings import EmbeddingModel
from app.llmclient import LLMClient
from app.vector_store import VectorStore
from app.promptbuilder import PromptBuilder

# RAGOrchestrator is the main orchestrator class that coordinates the vector store,
# prompt builder, and LLM client to execute the RAG process. It takes a user query, retrieves relevant documents, builds a prompt, and generates an answer using the LLM.
class RAGOrchestrator:
    
    def __init__(self, vector_store, prompt_builder, llm_client):
        self.vector_store = vector_store
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def run(self, query: str, k: int = 10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        docs = self.vector_store.search(query, k)

        if not docs:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        prompt = self.prompt_builder.build_prompt(query, docs)

        
        answer = self.llm_client.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {"title": d[0], "description": d[1]}
                for d in docs
            ]
        }