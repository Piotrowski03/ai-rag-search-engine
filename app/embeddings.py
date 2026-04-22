from sentence_transformers import SentenceTransformer
import torch
import time
import logging

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",batch_size: int=32):

        self.model = SentenceTransformer(model_name)

        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        self.logger.setLevel(logging.INFO)

        self.handler = logging.StreamHandler()

        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        self.handler.setFormatter(self.formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.handler)
        

    def encode(self, texts: list[str]):
        start_time = time.perf_counter()

        if not isinstance(texts, list):
            raise TypeError("Data is not a list")
        
        if not len(texts) > 0:
            raise ValueError("List is empty")
        
        if not  all(isinstance(t,str) for t in texts):
            raise TypeError("The List contains other data types than string")
        
        embeddings = self.model.encode(texts, convert_to_tensor=True,batch_size=self.batch_size)
        embeddings = torch.nn.functional.normalize(embeddings,dim = 1,p= 2)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        self.logger.info(f"Embedding completed | texts ={len(texts)} | time={elapsed_time:.4f}s")
        embeddings = embeddings.cpu().numpy().astype("float32")

        return embeddings
       
    