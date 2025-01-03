import numpy as np
from utils import knn_classify, calculate_metrics, load_dataset

def stratified_validation(k):
    print("Metodo scelto: Stratified Shuffle Split")
    
    # Chiediamo il numero di divisioni (K) per Stratified Shuffle Split
    while True:
        try:
            K = int(input("Inserisci il numero di divisioni (K): "))
            if K <= 0:
                print("Il numero di divisioni (K) deve essere maggiore di zero.")
                continue
            break
        except ValueError:
            print("Per favore, inserisci un numero valido per K.")
    
    print(f"Usando K = {K} divisioni")
    
    # Carichiamo il dataset
    X, y = load_dataset()
    
    # Percentuale fissa di divisione tra train e test (80/20)
    test_size = 0.2
    splits = stratified_shuffle_split(X, y, K, test_size)
    
    metrics_list = []
    
    for X_train, y_train, X_test, y_test in splits:
        predictions = knn_classify(X_train, y_train, X_test, k)
        metrics = calculate_metrics(y_test, predictions)
        metrics_list.append(metrics)
    
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    
    print("\nRisultati - Stratified Shuffle Split:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

def stratified_shuffle_split(X, y, K, test_size):
    # Prepariamo la divisione stratificata
    splits = []
    unique_classes = np.unique(y)
    
    # Per ogni classe, divideremo gli esempi in train e test
    for _ in range(K):
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            # Troviamo gli indici per la classe corrente
            class_indices = np.where(y == cls)[0]
            np.random.shuffle(class_indices)
            
            # Calcoliamo il numero di esempi per il test set
            test_size_count = int(len(class_indices) * test_size)
            test_indices.extend(class_indices[:test_size_count])
            train_indices.extend(class_indices[test_size_count:])
        
        # Otteniamo i dati di addestramento e test
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Aggiungiamo la divisione alla lista degli splits
        splits.append((X_train, y_train, X_test, y_test))
    
    return splits
