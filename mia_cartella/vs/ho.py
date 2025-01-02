import pandas as pd
from funzioni import knn_classify
import numpy as np

def holdout_process(k):
    print(df.dtypes)
    while True:
        scelta = input("Quale percentuale vuoi inserire training o test: ").upper()
        if scelta == "TRAINING":
            perc_training = float(input("Inserisci la percentuale per la suddivisone del training: "))
            numero_righe_training = int(len(df) * (perc_training / 100))
            training_set = df.iloc[:numero_righe_training]
            test_set = df.iloc[numero_righe_training:]
            break

        elif scelta == "TEST":
            perc_test = float(input("Inserisci la percentuale per la suddivisone del test (80=80%, 0.8=0.8%): "))
            numero_righe_test = int(len(df) * (perc_test / 100))
            test_set = df.iloc[:numero_righe_test]
            training_set = df.iloc[numero_righe_test:]
            break

        else:
            print("Valore non valido, va inserito training o test (non case-sensitive).")

    
    
    
    X_train = training_set.drop(columns=['classtype_v1', 'Sample code number']).values  # Feature di training
    y_train = training_set['classtype_v1'].values  # Target di training
    X_test = test_set.drop(columns=['classtype_v1', 'Sample code number' ]).values  # Feature di test


    predictions = knn_classify(X_train, y_train, X_test, k)
    
    return predictions