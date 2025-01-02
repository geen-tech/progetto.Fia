import numpy as np
import pandas as pd

def load_dataset():
    try:
        data = pd.read_csv("version_1.csv")
    except Exception as e:
        print("Errore nel caricamento del file:", e)
        exit()
    data = data.dropna()
    data['Single Epithelial Cell Size'] = data['Single Epithelial Cell Size'].str.strip()  # Rimuove gli spazi
    data['Single Epithelial Cell Size'] = data['Single Epithelial Cell Size'].str.replace(',', '').str.replace('€', '')
    data['Single Epithelial Cell Size'] = pd.to_numeric(data['Single Epithelial Cell Size'], errors='coerce')
    data['Bland Chromatin'] = data['Bland Chromatin'].str.strip()  # Rimuove gli spazi
    data['Bland Chromatin'] = data['Bland Chromatin'].str.replace(',', '').str.replace('€', '')
    data['Bland Chromatin'] = pd.to_numeric(data['Bland Chromatin'], errors='coerce')
    #data = data.select_dtypes(include=[np.number])
    X = data.drop(columns=["classtype_v1", "Sample code number","Blood Pressure","Heart Rate"]).values
    y = np.where(data["classtype_v1"].values == 2, 0, 1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def knn_classify(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        neighbors = np.argsort(distances)[:k]
        neighbor_labels = y_train[neighbors]
        predicted_label = np.bincount(neighbor_labels).argmax()
        predictions.append(predicted_label)
    return np.array(predictions)

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

def stratified_shuffle_split(X, y, K, test_size):
    from collections import Counter
    splits = []
    for _ in range(K):
        train_indices = []
        test_indices = []
        class_counts = Counter(y)
        test_class_counts = {key: int(value * test_size) for key, value in class_counts.items()}
        
        for class_label, count in test_class_counts.items():
            class_indices = np.where(y == class_label)[0]
            np.random.shuffle(class_indices)
            test_indices.extend(class_indices[:count])
            train_indices.extend(class_indices[count:])
        
        splits.append((X[train_indices], y[train_indices], X[test_indices], y[test_indices]))
    return splits