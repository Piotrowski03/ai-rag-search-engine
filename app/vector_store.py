from embeddings import EmbeddingModel
import torch
import faiss
class VectorStore:
    def __init__(self):
        self.threshold = 0.3
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.dimension = None
        self.texts = []
    def build_index(self, embeddings , texts):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().astype("float32")
        if embeddings is None or len(embeddings.shape) != 2 or embeddings.shape[0] == 0:
            raise ValueError("wrong input file")
        if not isinstance(texts, list):
            raise TypeError("Data is not a list")
        if not len(texts) > 0:
            raise ValueError("List is empty")
        if not  all(isinstance(t,str) for t in texts):
            raise TypeError("The List contains other data types than string")
        if len(embeddings) != len(texts):
            raise ValueError("Arguments are diffrent sizes")
        self.texts = texts
            
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def search(self, query, k):
        if self.index is None:
            raise ValueError("Index not built")
        if isinstance(query, str):
            query = [query]
        elif isinstance(query,list):
            if len(query) == 0 or not all(isinstance(q, str) for q in query):
                raise TypeError("Wrong query format")      
        else:
            raise TypeError("Wrong query format") 
        
        embedded_query = self.embedding_model.encode(query)
        embedded_query = embedded_query.cpu().numpy().astype("float32")
        distances, indices = self.index.search(embedded_query, k)
        
        all_results = []
        for q_idx, idx_row in enumerate(indices):
            results = []
            for rank, (i, score) in enumerate(zip(idx_row,distances[q_idx])):
                if score > self.threshold:
                    results.append({
                        "text" : self.texts[i],
                        "score" : float(f"{score:.4f}"),
                        "rank" : rank,
                        "query_id" : q_idx
                    })
        
            all_results.append(results)
        return all_results
