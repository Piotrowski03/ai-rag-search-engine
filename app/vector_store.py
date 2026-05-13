from app.embeddings import EmbeddingModel
from database.db_connection import DBConnector
from dotenv import load_dotenv

# VectorStore is responsible for storing and retrieving document embeddings.
# It uses the EmbeddingModel to convert queries into embeddings and interacts
# with the database to perform similarity searches based on those embeddings.
class VectorStore:
    def __init__(self, dotenv_path):
        """ Setting a default threshold for search results and 
        initializing the embedding model"""
        self.embedding_model = EmbeddingModel()
        self.db_connector = DBConnector(dotenv_path)
    def search(self, query, k = 10):
        if not isinstance(query, str)or not query.strip():
            raise TypeError("Wrong query format, query should be a non-empty string")
        query_embedding = self.embedding_model.encode([query])
        sql_query = """
                    SELECT title, description
                    FROM books
                    WHERE embedding <=> %s::vector < 0.7
                    ORDER BY embedding <=> %s::vector < 0.7 ASC
                    LIMIT %s
                    """
        results = self.db_connector.select_query(sql_query, (query_embedding[0].tolist(),query_embedding[0].tolist(), k))
        return results

