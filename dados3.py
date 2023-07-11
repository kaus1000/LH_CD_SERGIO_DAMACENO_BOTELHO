import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# Carregue os dados de treinamento e teste em DataFrames Pandas
start_time = time.time()
print("Carregando os dados de treinamento e teste...")
df_train1 = pd.read_csv('cars_train.csv', encoding='utf16', delimiter='\t')
df_test1 = pd.read_csv('cars_test.csv', encoding='utf16', delimiter='\t')
print("Tempo de execução:", time.time() - start_time, "segundos")

# Crie DataFrames Dask a partir dos DataFrames Pandas
start_time = time.time()
print("Criando os DataFrames Dask...")
df_train = dd.from_pandas(df_train1, npartitions=4)
df_test = dd.from_pandas(df_test1, npartitions=4)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Combine os dados de treinamento e teste
start_time = time.time()
print("Combinando os dados de treinamento e teste...")
df_combined = dd.concat([df_train, df_test])
print("Tempo de execução:", time.time() - start_time, "segundos")

# Converta todas as colunas em dados categóricos conhecidos
start_time = time.time()
print("Convertendo as colunas em dados categóricos...")
categorical_columns = ['marca', 'modelo', 'versao', 'cambio', 'tipo', 'cor']
df_combined = df_combined[categorical_columns].categorize()
print("Tempo de execução:", time.time() - start_time, "segundos")

# Use DummyEncoder do dask-ml para aplicar one-hot encoding
start_time = time.time()
print("Aplicando one-hot encoding...")
encoder = DummyEncoder()
df_combined_encoded = encoder.fit_transform(df_combined)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Concatene os dados codificados com as colunas restantes
start_time = time.time()
print("Concatenando os dados codificados...")
df_combined_final = dd.concat([df_combined_encoded, df_combined.drop(categorical_columns, axis=1)], axis=1)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Separe novamente os dados de treinamento e teste
start_time = time.time()
print("Separando os dados de treinamento e teste...")
X = df_combined_final.compute(scheduler='single-threaded')
X_train = X.iloc[:len(df_train)]
X_test = X.iloc[len(df_train):]
print("Tempo de execução:", time.time() - start_time, "segundos")

# Separe os dados de treinamento em recursos (X) e variável de destino (y)
start_time = time.time()
print("Separando os dados de treinamento em recursos e variável de destino...")
y = df_train['preco'].compute(scheduler='single-threaded')
print("Tempo de execução:", time.time() - start_time, "segundos")

# Divida os dados de treinamento em treinamento e validação
start_time = time.time()
print("Dividindo os dados de treinamento em treinamento e validação...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.3, random_state=42)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Crie modelos de regressão: Árvore de Decisão, Random Forest e Gradient Boosting
start_time = time.time()
print("Criando modelos de regressão...")
model_decision_tree = DecisionTreeRegressor()
model_random_forest = RandomForestRegressor()
model_gradient_boosting = GradientBoostingRegressor()
print("Tempo de execução:", time.time() - start_time, "segundos")

# Lista para armazenar os resultados de MSE e R² de cada modelo
results = []

# Treine e avalie cada modelo
for model in [model_decision_tree, model_random_forest, model_gradient_boosting]:
    start_time = time.time()
    print("Treinando o modelo", model.__class__.__name__ + "...")
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    results.append((model.__class__.__name__, mse_train, mse_val, r2_train, r2_val))
    print("Tempo de execução:", time.time() - start_time, "segundos")

# Imprima os resultados
for model_name, mse_train, mse_val, r2_train, r2_val in results:
    print(f"{model_name}:")
    print("MSE (Treinamento):", mse_train)
    print("MSE (Validação):", mse_val)
    print("R² (Treinamento):", r2_train)
    print("R² (Validação):", r2_val)
    print()

# Faça previsões nos dados de teste usando o modelo de Gradient Boosting
start_time = time.time()
print("Fazendo previsões nos dados de teste...")
model = model_gradient_boosting
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Salve as previsões em um arquivo
start_time = time.time()
print("Salvando as previsões em um arquivo...")
df_result = pd.DataFrame({'id': df_test1['id'], 'preco': y_pred_test})
df_result.to_csv('predicted.csv', index=False)


# Análise das principais estatísticas da base de dados
print("Análise das principais estatísticas da base de dados:")
df_train1.describe(include='all').transpose()

# Gráfico das principais estatísticas descritivas
df_train1.hist(figsize=(12, 12), bins=20)
plt.tight_layout()
plt.show()
# print("Tempo de execução:", time.time() - start_time, "segundos")

# # Obter as importâncias das características para o modelo de Random Forest
# start_time = time.time()
# print("Analisando as importâncias das características...")
# importances_rf = model_random_forest.feature_importances_
# feature_importances = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Random Forest Importance': importances_rf
# })
# feature_importances = feature_importances.sort_values(by='Random Forest Importance', ascending=False)
# print(feature_importances)
# print("Tempo de execução:", time.time() - start_time, "segundos")

# # Plotar um gráfico de barras das importâncias das características
# plt.figure(figsize=(10, 6))
# plt.bar(feature_importances['Feature'], feature_importances['Random Forest Importance'])
# plt.xticks(rotation='vertical')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Feature Importances - Random Forest')
# plt.tight_layout()
# plt.show()
