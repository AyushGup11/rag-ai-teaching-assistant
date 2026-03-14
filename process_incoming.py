import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import time

def create_embedding(text_list):
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        
        # Truncate long texts in batch
        batch = [text[:8000] if len(text) > 8000 else text for text in batch]
        
        try:
            r = requests.post("http://localhost:11434/api/embed", json={
                "model": "bge-m3",
                "input": batch
            }, timeout=30)
            r.raise_for_status()
            all_embeddings.extend(r.json()["embeddings"])
            
        except Exception as e:
            print(f"  Batch error, processing individually...")
            for text in batch:
                try:
                    r = requests.post("http://localhost:11434/api/embed", json={
                        "model": "bge-m3",
                        "input": [text]
                    }, timeout=30)
                    all_embeddings.append(r.json()["embeddings"][0])
                except:
                    all_embeddings.append(None)
        
        time.sleep(0.1)
    
    return all_embeddings

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarities)
top_results = 3
mac_indx = similarities.argsort()[::-1][0:top_results]
#rint(mac_indx)
new_df = df.loc[mac_indx]
#print(new_df[["title", "number", "text"]])

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])