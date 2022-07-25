from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('bert-base-nli-mean-tokens')
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)