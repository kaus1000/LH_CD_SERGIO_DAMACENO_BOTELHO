import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from sklearn.impute import SimpleImputer

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

# Preencher valores ausentes com a média usando o SimpleImputer do scikit-learn
start_time = time.time()
print("Preenchendo valores ausentes...")
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Crie o modelo de regressão: Árvore de Decisão
start_time = time.time()
print("Criando modelo de regressão...")
model_decision_tree = DecisionTreeRegressor()
print("Tempo de execução:", time.time() - start_time, "segundos")

# Treine o modelo de Árvore de Decisão
start_time = time.time()
print("Treinando o modelo de Árvore de Decisão...")
model_decision_tree.fit(X_train, y_train)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Faça previsões nos dados de treinamento e validação
start_time = time.time()
print("Fazendo previsões nos dados de treinamento e validação...")
y_pred_train = model_decision_tree.predict(X_train)
y_pred_val = model_decision_tree.predict(X_val)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Avalie o modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_val = mean_squared_error(y_val, y_pred_val)
r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)

# Imprima as métricas de avaliação
print("Árvore de Decisão:")
print("MSE (Treinamento):", mse_train)
print("MSE (Validação):", mse_val)
print("R² (Treinamento):", r2_train)
print("R² (Validação):", r2_val)

# Faça previsões nos dados de teste usando o modelo de Árvore de Decisão
start_time = time.time()
print("Fazendo previsões nos dados de teste...")
y_pred_test = model_decision_tree.predict(X_test)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Salve as previsões em um arquivo
start_time = time.time()
print("Salvando as previsões em um arquivo...")
df_result = pd.DataFrame({'id': df_test1['id'], 'preco': y_pred_test})
df_result.to_csv('predicted.csv', index=False)
print("Tempo de execução:", time.time() - start_time, "segundos")

# Obter as importâncias das características para o modelo de Árvore de Decisão
importances_dt = model_decision_tree.feature_importances_

# Criar DataFrame com as importâncias das características
feature_importances = pd.DataFrame({
    'Feature': df_combined_final.columns,
    'Decision Tree Importance': importances_dt
}).sort_values(by='Decision Tree Importance', ascending=False)


# Selecionar as características desejadas
selected_features = ['num_fotos', 'marca', 'modelo', 'versao', 'ano_de_fabricacao', 'ano_modelo',
                     'hodometro', 'cambio', 'num_portas', 'tipo', 'blindado', 'cor', 'tipo_vendedor',
                     'cidade_vendedor', 'estado_vendedor', 'entrega_delivery', 'troca', 'elegivel_revisao',
                     'dono_aceita_troca', 'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago',
                     'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda', 'veiculo_alienado']

# Filtrar o DataFrame com as características selecionadas
feature_importances_selected = feature_importances[feature_importances['Feature'].isin(selected_features)]

# Imprimir as importâncias das características
print(feature_importances_selected)

# Plotar um gráfico de barras das importâncias das características
plt.figure(figsize=(10, 6))
plt.bar(feature_importances_selected['Feature'], feature_importances_selected['Decision Tree Importance'])
plt.xticks(rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances - Decision Tree')
plt.tight_layout()
plt.show()
