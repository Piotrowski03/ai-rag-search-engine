from sentence_transformers import SentenceTransformer
import torch
import time
import logging
from typing import List

class EmbeddingModel:
    #constructor with default model and batch size, also setting up logging
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",batch_size: int=32):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        # Set the batch size for embedding
        self.batch_size = batch_size
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Create a stream handler for logging
        self.handler = logging.StreamHandler()
        # Define a formatter for the log messages
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # Apply the formatter to the handler
        self.handler.setFormatter(self.formatter)
        # Add the handler to the logger if it doesn't already have handlers
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.handler)
        
    #method to embed a list of texts
    def encode(self, texts: List[str]):
        start_time = time.perf_counter()
        # Validating input
        if not isinstance(texts, list):
            raise TypeError("Data is not a list")
        if not len(texts) > 0:
            raise ValueError("List is empty")
        if not  all(isinstance(t,str) for t in texts):
            raise TypeError("The List contains other data types than string")
        
        # Creating embeddings using the model
        embeddings = self.model.encode(texts, convert_to_tensor=True,batch_size=self.batch_size)
        embeddings = torch.nn.functional.normalize(embeddings,dim = 1,p= 2)
        end_time = time.perf_counter()
        # Calculating elapsed time and logging the embedding process
        elapsed_time = end_time - start_time
        self.logger.info(f"Embedding completed | texts ={len(texts)} | time={elapsed_time:.4f}s")
        embeddings = embeddings.cpu().numpy().astype("float32")

        return embeddings
       
    