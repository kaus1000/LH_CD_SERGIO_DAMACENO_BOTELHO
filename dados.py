# Importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Carregue os dados de treinamento e teste
df_train = pd.read_csv('cars_train.csv', encoding='utf16', delimiter='\t')
df_test = pd.read_csv('cars_test.csv', encoding='utf-16', delimiter='\t')

# Preencher com um valor constante, como 'N/A'
df_train['veiculo_alienado'] = df_train['veiculo_alienado'].fillna('N/A')
df_test['veiculo_alienado'] = df_test['veiculo_alienado'].fillna('N/A')

# Remova colunas desnecessárias
df_train = df_train.drop(['anunciante'], axis=1)
df_test = df_test.drop(['anunciante'], axis=1)

# Separe os dados de treinamento em recursos (X) e variável de destino (y)
X = df_train.drop('preco', axis=1)
y = df_train['preco']

# Combine os dados de treinamento e teste para aplicar a codificação one-hot encoding
df_combined = pd.concat([X, df_test])

# Codifique variáveis categóricas usando one-hot encoding
df_combined = pd.get_dummies(df_combined)

# Separe novamente os dados de treinamento e teste
X = df_combined[:len(df_train)]
X_test = df_combined[len(df_train):]

# Lide com valores ausentes usando uma estratégia de imputação (preenchendo com a média)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Crie um modelo de árvore de decisão
# model = DecisionTreeRegressor()

# Ou, se preferir, crie um modelo de regressão por floresta aleatória
model = RandomForestRegressor()


# Treine o modelo
model.fit(X, y)

# Faça previsões nos dados de treinamento
y_pred_train = model.predict(X)


# Calcule o MSE (Mean Squared Error)
mse = mean_squared_error(y, y_pred_train)
print('MSE:', mse)


# Faça previsões nos dados de teste
y_pred_test = model.predict(X_test)

# Salve as previsões em um arquivo
df_result = pd.DataFrame({'id': df_test['id'], 'preco': y_pred_test})
df_result.to_csv('predicted.csv', index=False)

