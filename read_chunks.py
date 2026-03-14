import requests
import os
import json
import pandas as pd
import numpy as np  
import time
import joblib
from sklearn.metrics.pairwise import cosine_similarity

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

jsons = [f for f in os.listdir("jsons") if f.endswith('.json')]
my_dicts = []
chunk_id = 0

for json_file in jsons:
    try:
        with open(f"jsons/{json_file}", encoding='utf-8') as f:
            content = json.load(f)
        
        print(f"Creating Embeddings for {json_file}")
        
        texts = [c.get('text', '') for c in content['chunks']]
        embeddings = create_embedding(texts)
        
        for i, chunk in enumerate(content['chunks']):
            if embeddings[i] is not None:
                chunk['chunk_id'] = chunk_id
                chunk['embedding'] = embeddings[i]
                chunk_id += 1
                my_dicts.append(chunk)     
    
    except Exception as e:
        print(f"  Error: {e}")
        continue

df = pd.DataFrame.from_records(my_dicts)

joblib.dump(df, 'embeddings.joblib')



