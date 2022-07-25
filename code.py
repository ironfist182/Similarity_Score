import uvicorn
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class similarity(BaseModel):
  sentence1: str
  sentence2: str

app = FastAPI()
pick = open('model_pkl' , 'rb')
model = lr = pickle.load(pick)

@app.get('/')
def index():
  return {'message':'Hello'}

@app.post('/Calculate_Similarity')
def calculate_sim(data:similarity):
  data = data.dict()
  sen1 = data['sentence1']
  sen2 = data['sentence2']
  sentences = [sen1,sen2]
  sentence_embeddings = model.encode(sentences)
  result = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
  return {'Similarity Score': str(result[0][0])}

if __name__ == '__main__':
  uvicorn.run(app,host='127.0.0.1',port=8000)