from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import *

# Carrega os embeddings e labels
embeddings = np.load('embeddings128.npy', allow_pickle=True)
file_path = 'cstnews'
texts, categories = load_data(file_path)


# Divide os embeddings em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(embeddings, categories, test_size=0.1)


# Inicializa e treina o classificador SVM com kernel linear
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = svm_classifier.predict(X_test)

# Avalia o desempenho do classificador
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print("SVM:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
