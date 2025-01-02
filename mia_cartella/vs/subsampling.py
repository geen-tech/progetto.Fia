import pandas as pd
import random
import numpy as np

def euclidean_distance(point1, point2):
    """Calcola la distanza euclidea tra due punti."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def classify_point(train_data, train_labels, test_point, k):
    """
    Classifica un punto di test in base ai k vicini più vicini nel set di training.
    """
    distances = []
    for index, train_point in train_data.iterrows():
        dist = euclidean_distance(train_point.values, test_point.values)
        distances.append((dist, train_labels.iloc[index]))
    
    # Ordina per distanza e seleziona i primi k vicini
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Conta le etichette dei vicini
    label_counts = {}
    for _, label in k_nearest:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Ritorna l'etichetta con il massimo conteggio
    return max(label_counts, key=label_counts.get)

def random_subsampling(k):
    print("Hai scelto Random Subsampling.")

    # Chiedi all'utente di specificare K grande (numero di esperimenti)
    while True:
        try:
            K_grande = int(input("Per favore, inserisci il numero di esperimenti (K grande): "))
            if K_grande > 0:
                break
            else:
                print("Inserisci un numero positivo.")
        except ValueError:
            print("Valore non valido. Inserisci un numero intero.")

    # Leggi il file version_1.csv e trasformalo in un DataFrame
    try:
        df = pd.read_csv("version_1.csv")
        print("Il dataset è stato caricato con successo.")

        # Escludi le righe con valori NaN inutili
        df = df.dropna()
        df = df.drop(columns=["Blood Pressure","Heart Rate"])
        df['Single Epithelial Cell Size'] = df['Single Epithelial Cell Size'].str.strip()  # Rimuove gli spazi
        df['Single Epithelial Cell Size'] = df['Single Epithelial Cell Size'].str.replace(',', '').str.replace('€', '')
        df['Single Epithelial Cell Size'] = pd.to_numeric(df['Single Epithelial Cell Size'], errors='coerce')
        df['Bland Chromatin'] = df['Bland Chromatin'].str.strip()  # Rimuove gli spazi
        df['Bland Chromatin'] = df['Bland Chromatin'].str.replace(',', '').str.replace('€', '')
        df['Bland Chromatin'] = pd.to_numeric(df['Bland Chromatin'], errors='coerce')
        #print("Righe con valori NaN rimosse.")

        print("Anteprima del dataset:")
        print(df.head())
    except FileNotFoundError:
        print("Errore: Il file version_1.csv non è stato trovato nella directory corrente.")
        return
    except Exception as e:
        print(f"Errore durante il caricamento del file: {e}")
        return

    # Verifica che ci siano abbastanza dati per il subsampling
    if len(df) < 2:
        print("Errore: il dataset deve contenere almeno due righe.")
        return

    # Mappa le etichette 4 → maligno, 2 → benigno
    df['classtype_v1'] = df['classtype_v1'].map({4: 'maligno', 2: 'benigno'})

    # Separazione delle feature e della variabile target
    try:
        X = df.drop(columns=["classtype_v1"])  # Feature
        y = df["classtype_v1"]  # Etichette
    except KeyError:
        print("Errore: il dataset deve contenere una colonna chiamata 'classtype_v1'.")
        return

    # Esegui il Random Subsampling
    print("\nEsecuzione del Random Subsampling...")
    accuracies = []

    for i in range(K_grande):
        # Dividi casualmente il dataset in train e test
        test_indices = random.sample(range(len(df)), int(0.3 * len(df)))  # 30% per il test
        train_indices = [idx for idx in range(len(df)) if idx not in test_indices]
        
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

        # Classifica ogni punto del test e calcola l'accuratezza
        correct_predictions = 0
        for idx, test_point in X_test.iterrows():
            predicted_label = classify_point(X_train, y_train, test_point, k)
            actual_label = y_test.loc[idx]
            if predicted_label == actual_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
        print(f"Esperimento {i + 1}: Accuratezza = {accuracy:.2f}")

    # Calcola la media delle accuratezze
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nMedia delle accuratezze su {K_grande} esperimenti: {avg_accuracy:.2f}")