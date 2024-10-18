from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from tqdm import tqdm
import os


# Caminho para a pasta cstnews
data_dir = 'cstnews'

# Leitura dos arquivos CSV
dataframes = []
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath, delimiter=';', quotechar='"')  # Ajustado para delimitador e aspas
            # Verificar se as colunas esperadas existem
            if 'documento' in df.columns and 'classe' in df.columns:
                dataframes.append(df[['documento', 'classe']])  # Selecionar apenas as colunas necessárias
            else:
                print(f"Arquivo {filename} não possui as colunas necessárias.")
        except Exception as e:
            print(f"Erro ao ler o arquivo {filename}: {e}")

# Combinar todos os DataFrames
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    print(data.info())  # Verificar se os dados foram carregados corretamente
else:
    raise ValueError("Nenhum arquivo CSV foi carregado com sucesso.")

# Preparação dos dados
texts = data['documento'].tolist()
labels = data['classe'].tolist()

# Pré-processamento dos textos
def preprocess_text(texts):
    # Exemplo simples de pré-processamento; ajuste conforme necessário
    return [text.lower() for text in texts]

preprocessed_texts = preprocess_text(texts)
print(f"Exemplo de textos pré-processados: {preprocessed_texts[:5]}")  # Verificar algumas amostras

# Dividir dados em treino, validação e teste
train_texts, temp_texts, train_labels, temp_labels = train_test_split(preprocessed_texts, labels, test_size=0.1, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)
print(f"Tamanhos dos dados: {len(train_texts)}, {len(val_texts)}, {len(test_texts)}")

# Tokenizar os textos usando BERT tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)
    print("Tokenizer carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o tokenizer: {e}")

def encode_texts(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_encodings = encode_texts(train_texts)
val_encodings = encode_texts(val_texts)
test_encodings = encode_texts(test_texts)

print(f"Tamanhos dos encodings: {len(train_encodings['input_ids'])}, {len(val_encodings['input_ids'])}, {len(test_encodings['input_ids'])}")

# Criar datasets
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

print(f"Tamanhos dos datasets: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

# Definir o modelo BERT para classificação de sequência
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)

# Parâmetros de treinamento
batch_size = 16
epochs = 4
learning_rate = 2e-5

# DataLoader para treino, validação e teste
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Configuração do otimizador, scheduler e função de perda
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Verificar se CUDA está disponível
print(f"CUDA disponível: {torch.cuda.is_available()}")

# Treinamento do modelo BERT
model.train()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': total_loss / len(progress_bar)})
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Avaliação no conjunto de validação
    model.eval()
    val_loss = 0
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1-score: {val_f1}")

# Avaliação do modelo no conjunto de teste
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating', leave=True):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Avaliação de desempenho
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='binary')

print("Bert:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
