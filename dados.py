# Importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

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

# Divida os dados em conjunto de treinamento e teste
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie um objeto de modelo de regressão linear
model = LinearRegression()

# Treine o modelo com o conjunto de treinamento
model.fit(X_train, y_train)

# Faça previsões no conjunto de treinamento e validação
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Calcule o MSE no conjunto de treinamento e validação
mse_train = mean_squared_error(y_train, y_pred_train)
mse_val = mean_squared_error(y_val, y_pred_val)

print('MSE (treinamento):', mse_train)
print('MSE (validação):', mse_val)

# Faça previsões nos dados de teste
y_pred_test = model.predict(X_test)

# Salve as previsões em um arquivo
df_result = pd.DataFrame({'id': df_test['id'], 'preco': y_pred_test})
df_result.to_csv('predicted.csv', index=False)
