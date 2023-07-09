import pandas as pd
import numpy as np

# Carregar o conjunto de dados
df = pd.read_csv('cars_train.csv', encoding='utf16',  delimiter='\t')

# Identificar as variáveis numéricas e categóricas
variaveis_numericas = df.select_dtypes(include=[np.number]).columns
variaveis_categoricas = df.select_dtypes(include=[np.object_]).columns

# Calcular estatísticas descritivas para as variáveis numéricas
estatisticas_numericas = df[variaveis_numericas].describe()

# Calcular a moda para as variáveis categóricas
moda_categoricas = df[variaveis_categoricas].mode().transpose()

# Imprimir as estatísticas descritivas das variáveis numéricas
print("Estatísticas Descritivas - Variáveis Numéricas")
print(estatisticas_numericas)

# Imprimir a moda das variáveis categóricas
print("\nModa - Variáveis Categóricas")
print(moda_categoricas)
