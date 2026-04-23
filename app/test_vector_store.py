from embeddings import EmbeddingModel
from vector_store import VectorStore

def main():
    # 🔹 przykładowe dane
    texts = [
        "FAISS is a library for efficient similarity search",
        "Machine learning is a field of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are used in deep learning",
        "Embeddings represent text as vectors"
    ]

    # inicialization of embedding model and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    #creatintg embeddings
    print("Building embeddings...")
    embeddings = embedding_model.encode(texts)

    # building FAISS index
    print("Building FAISS index...")
    vector_store.build_index(embeddings, texts)

    #example_queries
    queries = [
        "What is FAISS?",
        "Explain machine learning",
        "What are embeddings?"
    ]

    # searching for similar texts
    print("\n--- SEARCH RESULTS ---\n")
    results = vector_store.search(queries, k=3)

    # displaying results
    for q_idx, res in enumerate(results):
        print(f"\nQuery {q_idx + 1}: {queries[q_idx]}")
        print("-" * 40)

        if not res:
            print("No results above threshold")
            continue

        for r in res:
            print(f"Rank: {r['rank']}")
            print(f"Score: {r['score']}")
            print(f"Text: {r['text']}")
            print()

if __name__ == "__main__":
    main()