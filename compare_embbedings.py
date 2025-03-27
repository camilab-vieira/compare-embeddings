# ======================================================================================================================================================
# Importações
# ======================================================================================================================================================
import time
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ======================================================================================================================================================
# Configurações globais
# ======================================================================================================================================================
RESULTS_DIR = Path('Results')
EMBEDDINGS_DIR = RESULTS_DIR / 'Embeddings'
PLOTS_DIR = RESULTS_DIR / 'Plots'
MODELS_DIR = RESULTS_DIR / 'Models'
METRICS_DIR = RESULTS_DIR / 'Metrics'
EMBEDDINGS_FILE = EMBEDDINGS_DIR / 'embeddings.pkl'
RESULTS_FILE = METRICS_DIR / 'results.pkl'


for dir in [RESULTS_DIR, EMBEDDINGS_DIR, PLOTS_DIR, MODELS_DIR, METRICS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)


EMBEDDINGS = {
    # Inglês
    # "BART": "facebook/bart-base",
    # "BERT": "google-bert/bert-base-uncased",
    # "ROBERTA": "FacebookAI/roberta-base",
    ## "T5": "google-t5/t5-base",

    # Português
    # "BERTIMBAU": "neuralmind/bert-base-portuguese-cased", 
    ## "PTT5": "unicamp-dl/ptt5-v2-base",
    # "TUCANO": "TucanoBR/Tucano-630m",
    # "TEENNYTINYLLAMA": "nicholasKluge/TeenyTinyLlama-460m",

    # Multilingue
    "MBART": "facebook/mbart-large-50",
    # "MBERT": "google-bert/bert-base-multilingual-uncased",
    # "XLM-ROBERTA": "FacebookAI/xlm-roberta-base",
    ## "MT5": "google/mt5-base",
}
CLASSIFIERS = {
    "RFR": RandomForestClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42),
    "SVR": SVC(random_state=42),
}
HYPERPARAMETERS = {
    "RFR": {'bootstrap': [True, False], 'max_depth': [5, 10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 'n_estimators': [50, 100, 200], 'criterion': ['gini', 'entropy']},
    "MLP": {'activation': ['relu', 'logistic'], 'solver': ['adam', 'lbfgs']},
    "SVR": {'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100, 1000]},
}

# ======================================================================================================================================================
# Carregar datasets
# ======================================================================================================================================================
def load_data():
    train_url = "https://raw.githubusercontent.com/EduardoCavValenca/Automatic-Detection-of-Fake-News-in-Portuguese/main/data/csvs/train.csv"
    test_url = "https://raw.githubusercontent.com/EduardoCavValenca/Automatic-Detection-of-Fake-News-in-Portuguese/main/data/csvs/test.csv"
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)
    return df_train["content"], df_train["label"], df_test["content"], df_test["label"]

# ======================================================================================================================================================
# Calcular embedding 
# ======================================================================================================================================================
def calculate_embedding(texts, embedding_path, pooling='CLS'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_path)
    tokenizer.src_lang = "pt_XX"
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(embedding_path).to(device)
    model.eval()

    def get_embedding(text):
        start_time = time.time()
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
            outputs = model(**inputs)
            if pooling == 'mean':
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            else:
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        token_time = time.time() - start_time
        return embedding, token_time
    
    results  = Parallel(n_jobs=-1)(delayed(get_embedding)(text) for text in texts)
    embeddings, token_times = zip(*results)
    return np.array(embeddings), np.array(token_times)
    
# ======================================================================================================================================================
# Treinamento do Modelo 
# ======================================================================================================================================================
def train_and_optimize_model(X_train, y_train, X_test, y_test, model, param_grid):
    start_time = time.time()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    predictions = best_model.predict(X_test)

    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'f1_score': f1_score(y_test, predictions, average='macro'),
        'best_score': best_score,
        'training_time': elapsed_time
    }

    return best_model, metrics

# ======================================================================================================================================================
# Calcular Dificuldade das Instâncias 
# ======================================================================================================================================================
def instance_hardness(x_train, y_train, x_test, y_test, k=5):
  nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(x_train)
  _, indices = nbrs.kneighbors(x_test)
  neighbors = indices[:, 1:]
  diff_class = np.tile(y_test, (k, 1)).transpose() != y_train[neighbors]
  score = np.sum(diff_class, axis=1) / k
  return score

def plotar_ih_by_model(df_results, model):
  plt.figure(figsize=(10, 6))

  kdn_scores = np.sort(df_results['kdn_score'])
  cumulative_scores = np.arange(1, len(kdn_scores) + 1) / len(kdn_scores)
  plt.plot(kdn_scores, cumulative_scores, label="Overall")

  sorted_benign = np.sort(df_results[df_results['y_test'] == 0]['kdn_score'])
  cumulative_benign = np.arange(1, len(sorted_benign) + 1) / len(sorted_benign)
  plt.plot(sorted_benign, cumulative_benign, label="Benign class")

  sorted_malignant = np.sort(df_results[df_results['y_test'] == 1]['kdn_score'])
  cumulative_malignant = np.arange(1, len(sorted_malignant) + 1) / len(sorted_malignant)
  plt.plot(sorted_malignant, cumulative_malignant, label="Malignant class")

  plt.title(f'Cumulative Hardness - {model}')
  plt.xlabel('KDN Score')
  plt.ylabel('Cumulative Distribution')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(True)
  plt.xticks(np.arange(0, 1.1, 0.1))
  plt.yticks(np.arange(0, 1.1, 0.1))
  plt.savefig(PLOTS_DIR / f'cumulative_hardness_histogram_{model}.png', dpi=300)
  plt.close()

def plotar_ih(embeddings):
  plt.figure(figsize=(10, 6))

  for embedding_path, embedding in embeddings.items():
    kdn_scores = np.sort(embedding['kdn_score'])
    cumulative_scores = np.arange(1, len(kdn_scores) + 1) / len(kdn_scores)
    plt.plot(kdn_scores, cumulative_scores, label=embedding_path)

  plt.title(f'Cumulative Hardness')
  plt.xlabel('KDN Score')
  plt.ylabel('Cumulative Distribution')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(True)
  plt.xticks(np.arange(0, 1.1, 0.1))
  plt.yticks(np.arange(0, 1.1, 0.1))
  plt.savefig(PLOTS_DIR / f'cumulative_hardness_histogram.png', dpi=300)
  plt.close()

# ======================================================================================================================================================
# Execução 
# ======================================================================================================================================================
def main():
    print("Starting...")

    X_train, y_train, X_test, y_test = load_data()

    embeddings = {}
    if EMBEDDINGS_FILE.exists():
        print("Processing embedding folder")
        with open(EMBEDDINGS_FILE, 'rb') as file:
            embeddings = pickle.load(file)
        print(embeddings.keys())
    results = {}
    if RESULTS_FILE.exists():
        print("Processing results folder")
        with open(RESULTS_FILE, 'rb') as file:
            results = pickle.load(file)
    
    for embedding_name, embedding_path in EMBEDDINGS.items():
        print(f"Processing with embedding: {embedding_name}")
        
        # Gerar embeddings
        if embedding_name not in embeddings.keys():
            X_train_embeddings, train_embeddings_time = calculate_embedding(X_train, embedding_path)
            X_test_embeddings, test_embeddings_time = calculate_embedding(X_test, embedding_path)
            kdn_score = instance_hardness(X_train_embeddings, np.array(y_train), X_test_embeddings, np.array(y_test))
        
            embeddings[embedding_name] = {
                'X_train_embeddings': X_train_embeddings,
                'train_embeddings_time': train_embeddings_time,
                'X_test_embeddings': X_test_embeddings,
                'test_embeddings_time': test_embeddings_time,
                'kdn_score': kdn_score
            }
            with open(EMBEDDINGS_DIR / 'embeddings.pkl', 'wb') as file:
                pickle.dump(embeddings, file)
        else:
            print(embeddings[embedding_name].keys())
            X_train_embeddings, X_test_embeddings, kdn_score = embeddings[embedding_name]['X_train_embeddings'], embeddings[embedding_name]['X_test_embeddings'], embeddings[embedding_name]['kdn_score']

        plotar_ih_by_model(pd.DataFrame({'y_test': np.array(y_test).flatten(),'kdn_score': kdn_score}), embedding_name)
        
        if not results.get(embedding_name):
            results[embedding_name] = {}
            print(f"Testing with embedding: {embedding_name}")
            def train_classifier(model_name, model):
                best_model, metrics = train_and_optimize_model(X_train_embeddings, np.array(y_train), X_test_embeddings, np.array(y_test), model, HYPERPARAMETERS[model_name])
                
                # Salvar modelo e resultados
                print(f"Finish: {embedding_name}_{model_name}")
                with open(MODELS_DIR / f"{embedding_name}_{model_name}.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                with open(METRICS_DIR / f"{embedding_name}_{model_name}.pkl", "wb") as f:
                    pickle.dump(metrics, f) 
                
                return model_name, metrics

            classifier_results = Parallel(n_jobs=-1)(delayed(train_classifier)(model_name, model) for model_name, model in CLASSIFIERS.items())
            results[embedding_name] = {model_name: metrics for model_name, metrics in classifier_results}
            with open(RESULTS_FILE, "wb") as f:
                pickle.dump(results, f)

    plotar_ih(embeddings)
    data = []
    for embedding_name, models in results.items():
        for model_name, metrics in models.items():
            metrics['embedding_name'] = embedding_name
            metrics['model_name'] = model_name
            metrics['token_time'] = np.sum(embeddings[embedding_name].get('train_embeddings_time', None)) + (np.sum(embeddings[embedding_name].get('test_embeddings_time', None)))
            data.append(metrics)

    df = pd.DataFrame(data, columns=['embedding_name', 'model_name', 'token_time',  'training_time', 'accuracy', 'f1_score', 'best_score', 'confusion_matrix'])
    df.to_csv(METRICS_DIR / 'results.csv', index=False)

    print("Finished processing!")


if __name__ == "__main__":
    main()
