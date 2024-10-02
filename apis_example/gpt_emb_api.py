import os
from openai import OpenAI

with open("./apis/openai.key", 'r') as f:
    key = f.readlines()[0].strip()
    
client = OpenAI(api_key=key)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding