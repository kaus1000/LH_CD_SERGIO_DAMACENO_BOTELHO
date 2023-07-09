import pandas as pd
import numpy as np

# Carregar o conjunto de dados
df = pd.read_csv('cars_train.csv', encoding='utf16',  delimiter='\t')

# Calcular as estatísticas descritivas
estatisticas_descritivas = df.describe()

# Imprimir as estatísticas descritivas
print(estatisticas_descritivas)

media_hodometro = np.mean(df['hodometro'])
print("Média do hodômetro:", media_hodometro)

import matplotlib.pyplot as plt

plt.hist(df['preco'], bins=30)
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.title('Distribuição dos Preços')
plt.show()
