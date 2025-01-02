import pandas as pd
import numpy as np
# Funzione per calcolare la distanza euclidea
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

# Funzione per il classificatore k-NN
def knn_classify(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = euclidean_distance(X_train, test_point)
        k_indices = distances.argsort()[:k]  # Indici dei k vicini più vicini
        k_labels = y_train[k_indices]       # Label corrispondenti
        unique, counts = np.unique(k_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])  # Label più frequente
    return np.array(predictions)

