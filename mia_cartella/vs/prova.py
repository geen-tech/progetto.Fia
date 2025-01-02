import pandas as pd
df_grezzo = pd.read_csv("version_1.csv")
df_grezzo1 = df_grezzo.drop(columns=['Blood Pressure', 'Heart Rate'])
df = df_grezzo1.dropna()
df['Single Epithelial Cell Size'] = df['Single Epithelial Cell Size'].str.strip()  # Rimuove gli spazi
df['Single Epithelial Cell Size'] = df['Single Epithelial Cell Size'].str.replace(',', '').str.replace('€', '')
df['Single Epithelial Cell Size'] = pd.to_numeric(df['Single Epithelial Cell Size'], errors='coerce')
df['Bland Chromatin'] = df['Bland Chromatin'].str.strip()  # Rimuove gli spazi
df['Bland Chromatin'] = df['Bland Chromatin'].str.replace(',', '').str.replace('€', '')
df['Bland Chromatin'] = pd.to_numeric(df['Bland Chromatin'], errors='coerce')
#columns_to_display = ['Single Epithelial Cell Size', 'Bland Chromatin', 'classtype_v1']
columns_to_display = ['Bland Chromatin']
chunk_size = 20  # Numero di righe per blocco

for index, value in df['Bland Chromatin'].items():
    print(f"Riga {index}: {value} -> Tipo: {type(value)}")
'''
for i in range(0, len(df), chunk_size):
    print(df.iloc[i:i + chunk_size][columns_to_display])
    input("Premi Invio per continuare...")
'''
