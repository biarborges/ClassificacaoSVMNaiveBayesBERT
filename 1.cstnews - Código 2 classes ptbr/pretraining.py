import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from preprocessing import *

def vectorize_with_bertimbau(texts, max_length=128, batch_size=8):  # Reduzir o batch size
    model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    try:
        embeddings = np.load('embeddings128train.npy', allow_pickle=True)
        print("Embeddings carregados do arquivo.")
        print(embeddings.shape)
    except FileNotFoundError:
        print("Arquivo de embeddings não encontrado. Vetorizando textos...")

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Vetorizando textos"):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length, add_special_tokens=True, return_attention_mask=True)
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)

        np.save('embeddings128train.npy', embeddings)
        print("Embeddings salvos em embeddings128train.npy.")

    return embeddings

# Carrega e pré-processa os dados
file_path = 'cstnews'
texts, categories = load_data(file_path)
texts = preprocess_text(texts)

# Limitar o tamanho dos dados para teste
#subset_size = 10000  # Ajuste este valor para testar com diferentes tamanhos de dados
#texts = texts[:subset_size]

# Vetoriza os textos
embeddings = vectorize_with_bertimbau(texts)
