from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
model.save('offline_models/all-mpnet-base-v2')
