import pandas as pd
import os
import numpy as np

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste. Verifica il percorso.")
    df = pd.read_csv(file_path)
    print(f"Dataset caricato con {df.shape[0]} righe e {df.shape[1]} colonne.")
    print("Nomi delle colonne presenti nel dataset:")
    print(df.columns)
    return df

def preprocess_classes(df, class_column):
    # Mappa i valori numerici in etichette testuali
    mapping = {2: 'benign', 4: 'malignant'}
    df[class_column] = df[class_column].map(mapping)
    df = df.dropna(subset=[class_column])  # Rimuove righe con valori NaN
    print(f"Valori univoci nella colonna '{class_column}': {df[class_column].unique()}")
    return df

def manual_train_test_split(df, test_size=0.2):
    # Mescola il dataset
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calcola la dimensione del set di test
    test_len = int(len(shuffled_df) * test_size)
    
    # Divide in train e test
    test_set = shuffled_df.iloc[:test_len]
    train_set = shuffled_df.iloc[test_len:]
    
    print(f"Train set: {train_set.shape[0]} righe, Test set: {test_set.shape[0]} righe")
    return train_set, test_set

# Percorso del file
file_path = "version_1.csv"

try:
    df = load_dataset(file_path)
    class_column = 'classtype_v1'
    df = preprocess_classes(df, class_column)
    train_data, test_data = manual_train_test_split(df)

    print("\nEsempio train set:")
    print(train_data.head())

    print("\nEsempio test set:")
    print(test_data.head())

    # Salva i risultati
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    print("\nTrain e Test set salvati correttamente come train_data.csv e test_data.csv.")
except Exception as e:
    print(f"Errore: {e}")





