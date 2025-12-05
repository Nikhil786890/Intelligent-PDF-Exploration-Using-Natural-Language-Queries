#the pdf reader comes first which reads or extracts the pdf ,then comes the text splitter which splits the text taht was read into small chunks then comess
#the embedding or vector embedding which are random values or numbers given to each text and these are then saved to  and each vector or number corresponds to each chunk 
#which can be later saved to vector db like FAISSs
#the embedder here will convert the queries and the chunks into vectors so that they can be comapared for this we can use sentence transformers
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load model locally
model_path = os.path.join("offline_models", "all-mpnet-base-v2")
model = SentenceTransformer(model_path)

def get_embedding(text: str) -> np.ndarray:
    """Convert text into normalized vector embedding."""
    vec = model.encode([text])[0]
    return vec / np.linalg.norm(vec)