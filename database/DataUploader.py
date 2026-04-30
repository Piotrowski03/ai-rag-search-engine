
import os
import sys
from app.embeddings import EmbeddingModel
from database.DataPrep import DataPreparator
from database.db_connection import DBConnector
from pathlib import Path



if __name__ == "__main__":

     dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
     db_connector = DBConnector(dotenv_path)
     data_preparator = DataPreparator()
     embedding_model = EmbeddingModel()

     data = data_preparator.read_data(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "books_summary.csv"))
     descriptions = [d["summaries"] for d in data]
     embeddings = embedding_model.encode(descriptions)

     for d, emb in zip(data, embeddings):
         d["embedding"] = emb.tolist()
     db_connector.add_data(data)
