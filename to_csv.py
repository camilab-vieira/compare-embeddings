import pickle
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('Results')
METRICS_DIR = RESULTS_DIR / 'Metrics'
EMBEDDINGS_DIR = RESULTS_DIR / 'Embeddings'
RESULTS_FILE = METRICS_DIR / 'results.pkl'
EMBEDDINGS_FILE = EMBEDDINGS_DIR / 'embeddings.pkl'


with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

with open(EMBEDDINGS_FILE, 'rb') as f:
    embedding = pickle.load(f)

data = []
for embedding_name, models in results.items():
    for model_name, metrics in models.items():
        metrics['embedding_name'] = embedding_name
        metrics['model_name'] = model_name
        metrics['token_time'] = np.sum(embedding[embedding_name].get('train_embeddings_time', None)) + (np.sum(embedding[embedding_name].get('test_embeddings_time', None)))
        data.append(metrics)

df = pd.DataFrame(data, columns=['embedding_name', 'model_name', 'token_time',  'training_time', 'accuracy', 'f1_score', 'best_score', 'confusion_matrix'])
df.to_csv(METRICS_DIR / 'results.csv', index=False)
