import os
import pandas as pd

# Leitura dos arquivos CSV
def load_data (data_dir):
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

    return texts, labels

# Pré-processamento dos textos
def preprocess_text(texts):
    # Exemplo simples de pré-processamento; ajuste conforme necessário
    return [text.lower() for text in texts]