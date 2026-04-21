from sentence_transformers import SentenceTransformer
import torch
import time

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",batch_size: int=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size


    def encode(self, texts: list[str]):
        if not isinstance(texts, list):
            raise TypeError("Data is not a list")
        if not len(texts) > 0:
            raise ValueError("List is empty")
        if not  all(isinstance(t,str) for t in texts):
            raise TypeError("The List contains other data types than string")
        embeddings = self.model.encode(texts, convert_to_tensor=True,batch_size=self.batch_size)
        embeddings = torch.nn.functional.normalize(embeddings,dim = 1,p= 2)
        return embeddings
       
    