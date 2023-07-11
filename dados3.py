import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregue os dados de treinamento e teste em DataFrames Pandas
df_train1 = pd.read_csv('cars_train.csv', encoding='utf16', delimiter='\t')
df_test1 = pd.read_csv('cars_test.csv', encoding='utf16', delimiter='\t')

# Crie DataFrames Dask a partir dos DataFrames Pandas
df_train = dd.from_pandas(df_train1, npartitions=4)
df_test = dd.from_pandas(df_test1, npartitions=4)

# Combine os dados de treinamento e teste
df_combined = dd.concat([df_train, df_test])

# Converta as colunas relevantes em dados categóricos conhecidos
categorical_columns = ['marca', 'modelo', 'versao', 'cambio', 'tipo', 'cor']
df_combined = df_combined[categorical_columns].categorize()

# Use DummyEncoder do dask-ml para aplicar one-hot encoding
encoder = DummyEncoder()
df_combined_encoded = encoder.fit_transform(df_combined)

# Concatene os dados codificados com as colunas restantes
df_combined_final = dd.concat([df_combined_encoded, df_combined.drop(categorical_columns, axis=1)], axis=1)

# Separe novamente os dados de treinamento e teste
X = df_combined_final.compute(scheduler='single-threaded')
X_train = X.iloc[:len(df_train)]
X_test = X.iloc[len(df_train):]

# Separe os dados de treinamento em recursos (X) e variável de destino (y)
y = df_train['preco'].compute(scheduler='single-threaded')

# Divida os dados de treinamento em treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.3, random_state=42)

# Crie modelos de regressão: Árvore de Decisão, Random Forest e Gradient Boosting
model_decision_tree = DecisionTreeRegressor()
model_random_forest = RandomForestRegressor()
model_gradient_boosting = GradientBoostingRegressor()

# Lista para armazenar os resultados de MSE e R² de cada modelo
results = []

# Treine e avalie cada modelo
for model in [model_decision_tree, model_random_forest, model_gradient_boosting]:
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    results.append((model.__class__.__name__, mse_train, mse_val, r2_train, r2_val))

# Imprima os resultados
for model_name, mse_train, mse_val, r2_train, r2_val in results:
    print(f"{model_name}:")
    print("MSE (Treinamento):", mse_train)
    print("MSE (Validação):", mse_val)
    print("R² (Treinamento):", r2_train)
    print("R² (Validação):", r2_val)
    print()

# Faça previsões nos dados de teste usando o modelo de Gradient Boosting
model = model_gradient_boosting
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

# Salve as previsões em um arquivo
df_result = pd.DataFrame({'id': df_test1['id'], 'preco': y_pred_test})
df_result.to_csv('predicted.csv', index=False)

# Obter as importâncias das características para cada modelo
importances_dt = model_decision_tree.feature_importances_[:len(categorical_columns)]
importances_rf = model_random_forest.feature_importances_[:len(categorical_columns)]
importances_gb = model_gradient_boosting.feature_importances_[:len(categorical_columns)]

# Criar um DataFrame com as importâncias das características
feature_importances = pd.DataFrame({
    'Feature': categorical_columns,
    'Decision Tree Importance': importances_dt,
    'Random Forest Importance': importances_rf,
    'Gradient Boosting Importance': importances_gb
})

# Organizar as colunas por importância (Random Forest)
feature_importances = feature_importances.sort_values(by='Random Forest Importance', ascending=False)

# Imprimir as importâncias das características
print(feature_importances[['Feature', 'Random Forest Importance']])

# Plotar um gráfico de barras das importâncias das características
plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Random Forest Importance'])
plt.xticks(rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances - Random Forest')
plt.tight_layout()
plt.show()
