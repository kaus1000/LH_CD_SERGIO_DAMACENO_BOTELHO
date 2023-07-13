# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Manipulação de dados
import pandas as pd
import numpy as np
import os # accessing directory structure

#LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Normalização dos dados
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import MinMaxScaler

#Carregar o csv para Dataframe
df = pd.read_csv('cars_train.csv',encoding='utf16', delimiter='\t')


#Análise Exploratória dos Dados
# Total de linhas e colunas
print(df.shape)

#Primeiras linhas 
print(df.head())
#Amostra aleatória
print(df.sample(5))

print(df.info())

#ENRIQUECIMENTO DOS DADOS
#Colunas Novas
# Criando a variável 'idade_do_carro'
df['idade_do_carro'] = pd.to_datetime('today').year - df['ano_de_fabricacao']

# Criando a variável 'é_luxo'
marcas_de_luxo = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'Ferrari', 'Lamborghini'] # Adicione outras marcas conforme necessário
df['é_luxo'] = df['marca'].apply(lambda x: 1 if x in marcas_de_luxo else 0)
df['marca_modelo'] = df['marca'] + "_" + df['modelo']

df.head()
#Lista de itens da variavél dammy
categorical_columns = ['id', 'marca', 'modelo', 'versao', 'cambio', 'tipo', 
                       'blindado', 'cor', 'tipo_vendedor', 'cidade_vendedor', 
                       'estado_vendedor', 'anunciante', 'dono_aceita_troca',
                       'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago',
                       'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda','é_luxo','marca_modelo']
# for col in categorical_columns:
#     print(f'Column: {col}')
#     print(df[col].value_counts(normalize=True))
#     print()

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('BoxPlot')
sns.set_palette("pastel")
sns.boxplot(x='preco', y='cor', data=df, width=0.6, fliersize=2.5)
plt.title('BoxPlot - preco x cor'); plt.show()






#VERIFICAÇÃO DE DADOS FALTANTES
print('TOTAL Null\n')
print(df.isnull().sum(),'\n')
print('TOTAL NA\n')
print(df.isna().sum(),)


#Retirada da coluna vazia veiculo_alienado
columns_to_drop = ['veiculo_alienado']
df = df.drop(columns=columns_to_drop)
print(df.shape)





#AVALIAÇÃO
df.hist(bins=80, figsize=(18,12),color="red"); plt.show()
fig, axs = plt.subplots(6, 1, figsize=(8, 20))

# Plota um histograma para o preço
axs[0].hist(df['preco'], bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Distribuição do Preço', fontsize=10)
axs[0].set_xlabel('Preço',fontsize=8)
axs[0].set_ylabel('Frequência')
axs[0].grid(axis='y', alpha=0.75)

# Plota um histograma para o ano de fabricação
axs[1].hist(df['ano_de_fabricacao'], bins=30, color='skyblue', edgecolor='black')
axs[1].set_title('Distribuição do Ano de Fabricação' , fontsize=10)
axs[1].set_xlabel('Ano de Fabricação' , fontsize=8)
axs[1].set_ylabel('Frequência' , fontsize=8)
axs[1].grid(axis='y', alpha=0.75)

# Plota um gráfico de barras para os 10 estados com mais registros
top_states = df['estado_vendedor'].value_counts().nlargest(10)
top_states.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[2])
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=50,fontsize=6)
axs[2].set_title('10 Estados com mais Registros' , fontsize=10)
axs[2].set_xlabel('Estado' , fontsize=8)
axs[2].set_ylabel('Frequência' , fontsize=8)
axs[2].grid(axis='x', alpha=0.75)

# Plota um gráfico de barras para os tipos de transmissão
transmission_counts = df['cambio'].value_counts()
transmission_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[3])
axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=50, fontsize=6)
axs[3].set_title('Tipos de Transmissão' , fontsize=10)
axs[3].set_xlabel('Tipo de Transmissão', fontsize=6)
axs[3].set_ylabel('Frequência', fontsize=8)
axs[3].grid(axis='y', alpha=0.75)

# Plota um gráfico de barras para as 10 marcas com mais registros
top_brands = df['marca'].value_counts().nlargest(10)
top_brands.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[4])
axs[4].set_xticklabels(axs[4].get_xticklabels(), rotation=50, fontsize=6)
axs[4].set_title('10 Marcas com mais Registros', fontsize=10)
axs[4].set_xlabel('Marca', fontsize=8)
axs[4].set_ylabel('Frequência', fontsize=8)
axs[4].grid(axis='y', alpha=0.75)

# Calcula o preço médio por marca
mean_price_by_brand = df.groupby('marca')['preco'].mean()

# Plota o preço médio por marca
sns.barplot(x=mean_price_by_brand.index, y=mean_price_by_brand.values, ax=axs[5])
axs[5].set_xticklabels(axs[5].get_xticklabels(), rotation=45)
axs[5].set_xlabel('Marca', fontsize=10)
axs[5].set_ylabel('Preço Médio', fontsize=8)
axs[5].set_title('Preço Médio por Marca', fontsize=8)

# Ajusta o layout para melhor visualização
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()



#VERIFICAÇÃO DE OUTLIERS

# Hipótese 1: Carros de marcas populares são mais baratos do que os de outras marcas.
popular_brands = ['VOLKSWAGEN', 'CHEVROLET', 'FORD']
avg_price_popular_brands = df[df['marca'].isin(popular_brands)]['preco'].mean()
avg_price_other_brands = df[~df['marca'].isin(popular_brands)]['preco'].mean()

# Hipótese 2: Carros com transmissão automática são mais caros do que carros com outros tipos de transmissão.
avg_price_auto = df[df['cambio'] == 'Automática']['preco'].mean()
avg_price_other_trans = df[df['cambio'] != 'Automática']['preco'].mean()

# Hipótese 3: Carros que ainda estão na garantia de fábrica são mais caros do que aqueles que não estão.
avg_price_warranty = df[df['garantia_de_fábrica'] == 'Garantia de fábrica']['preco'].mean()
avg_price_no_warranty = df[df['garantia_de_fábrica'] != 'Garantia de fábrica']['preco'].mean()

# Pergunta de negócio 1: Qual é o melhor estado registrado na base de dados para vender um carro de marca popular e por quê?
df_popular_brands = df[df['marca'].isin(popular_brands)]
avg_price_state_popular_brands = df_popular_brands.groupby('estado_vendedor')['preco'].mean()
best_state_to_sell_popular_brand = avg_price_state_popular_brands.idxmax()

# Pergunta de negócio 2: Qual é o melhor estado para comprar uma picape com transmissão automática e por quê?
df_automatic_pickups = df[(df['cambio'] == 'Automática') & (df['tipo'] == 'Picape')]
avg_price_state_automatic_pickups = df_automatic_pickups.groupby('estado_vendedor')['preco'].mean()
best_state_to_buy_automatic_pickup = avg_price_state_automatic_pickups.idxmin()

# Pergunta de negócio 3: Qual é o melhor estado para comprar carros que ainda estão na garantia de fábrica e por quê?
df_warranty = df[df['garantia_de_fábrica'] == 'Garantia de fábrica']
avg_price_state_warranty = df_warranty.groupby('estado_vendedor')['preco'].mean()
best_state_to_buy_warranty = avg_price_state_warranty.idxmin()

{
    "avg_price_popular_brands": avg_price_popular_brands,
    "avg_price_other_brands": avg_price_other_brands,
    "avg_price_auto": avg_price_auto,
    "avg_price_other_trans": avg_price_other_trans,
    "avg_price_warranty": avg_price_warranty,
    "avg_price_no_warranty": avg_price_no_warranty,
    "best_state_to_sell_popular_brand": best_state_to_sell_popular_brand,
    "best_state_to_buy_automatic_pickup": best_state_to_buy_automatic_pickup,
    "best_state_to_buy_warranty": best_state_to_buy_warranty,
}





#TRANSFORMANDO CATEGORIAS EM VALORES
#Ajuste das variáveis categóricas para valores 0 e 1
df = pd.get_dummies(df, columns=categorical_columns, prefix=['id', 'marca', 'modelo', 'versao', 'cambio', 'tipo', 
                       'blindado', 'cor', 'tipo_vendedor', 'cidade_vendedor', 
                       'estado_vendedor', 'anunciante', 'dono_aceita_troca',
                       'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago',
                       'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda','é_luxo','marca_modelo'])


#CORRELAÇÃO ENTRE AS VARIÁVEIS
plt.figure(figsize=(9,7))
sns.set_theme()
sns.heatmap(df.corr(), annot=True, fmt=".1f",linewidth=.5 ,cmap="RdBu");plt.show()


# Correlação entre as variáveis e a váriavel alvo
df.corr()["preco"].sort_values(ascending = False)


#Separação dos dados em Treino e Teste

#DADOS DAS VARIÁVEIS INDEPENDENTES
X = df.drop(["preco"], axis = 1)

#DADOS DA VARIÁVEL DEPENDENTE, QUE QUEREMOS ENCONTRAR
y = df["preco"]

#DIVISÃO DO DATA FRAME EM TEST E TREINAMENTO 70% TREINAMENTO E 30% TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=30)

#Tamanho das Amostras de Treino e Teste
print(f"X_train dimensões {X_train.shape} e tamanho {X_train.size}")
print(f"X_test dimensões {X_test.shape} e tamanho {X_test.size}")
print(f"y_train dimensões {y_train.shape} e tamanho {y_train.size}")
print(f"y_test dimensões {y_test.shape} e tamanho {y_test.size}")

#Modelo LinearRegression()
# Criando o modelo LinearRegression()
regLinear = LinearRegression()
# Realizar treinamento do modelo
regLinear.fit(X_train, y_train)
# Realizar predição com os dados separados para teste
pred_regLinear = regLinear.predict(X_test)


#intercept
iRG = (regLinear.intercept_)
print("Intercept: "+str(iRG))
# Visualização dos 03 primeiros resultados
print(f'Predição amostra de 3: {pred_regLinear[:3]}')
#Mean absolute error
maeRG = mean_absolute_error(y_test, pred_regLinear)
print('Erro absoluto médio (MAE): %.2f' % maeRG)
#Mean squared error
mRG = mean_squared_error(y_test, pred_regLinear)
print('Erro quadrado médio (MSE): %.2f' % mRG)
# R2
r2RG = r2_score(y_test, pred_regLinear)
print('R2: %.6f' % r2RG)

#VALIDAÇÃO DO MODELO 1: CrossValidation
cv_score = np.sqrt(-cross_val_score(regLinear,X_test,pred_regLinear,cv=5,scoring='neg_mean_squared_error'))
cv1RG = cv_score.mean()
cv2RG = cv_score.std()
print("Média: %.6f" %cv1RG)
print("Desvio Padrão: %.6f" %cv2RG)
print("Shape:",pred_regLinear.shape)
print("Scores neg_mean_squared_error: ",cv_score)

y_test = np.array(y_test)



plt.figure(figsize=(15,5))
plt.plot(pred_regLinear[:100], linewidth=1.5, color='r')
plt.plot(y_test[:100], linewidth=1.2,color='b')
plt.title('Valores preditos x  Valores reais : Modelo Regressão Linear',size=18)
plt.legend(['Predições','Real'],fontsize=12)
plt.show()

#Modelo LinearRegression() com dados normalizados StandardScaler()
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
# Inicializa a regressão linear
rLinear = LinearRegression()
# Ajusta o modelo aos dados de treinamento (aprende os coeficientes)
rLinear.fit(X_train_scaler,y_train)
# Faz previsões nos dados de teste
y_pred_scaler = rLinear.predict(X_test_scaler)

# VISUALIZANDO OS RESULTADOS
# Interceptação
iRL = (rLinear.intercept_)
print("Interceptação: "+str(iRL))
# Visualização dos 3 primeiros resultados
print(f'Previsão amostra de 3: {y_pred_scaler[:3]}')
# Erro absoluto médio
maeRL = mean_absolute_error(y_test, pred_regLinear)
print('Erro Absoluto Médio (MAE): %.2f' % maeRL)
# Erro quadrático médio
mRL = mean_squared_error(y_test, y_pred_scaler)
print('Erro Quadrático Médio (MSE): %.2f' % mRL)
# R2
r2RL = r2_score(y_test, y_pred_scaler)
print('R2: %.6f' % r2RL)


# VALIDAÇÃO DO MODELO 2: CrossValidation

cv_score = np.sqrt(-cross_val_score(rLinear,X_test_scaler,y_pred_scaler,cv=5,scoring='neg_mean_squared_error'))
cv1RL = cv_score.mean()
cv2RL = cv_score.std()
print("Média: %.6f" %cv1RL)
print("Desvio Padrão: %.6f" %cv2RL)
print("Shape:",pred_regLinear.shape)
print("Scores neg_mean_squared_error: ",cv_score)

y_test = np.array(y_test)

plt.figure(figsize=(15,5))
plt.plot(y_pred_scaler[:100], linewidth=1.5, color='r')
plt.plot(y_test[:100], linewidth=1.2,color='b')
plt.title('Valores previstos x Valores reais: Modelo Regressão Linear com StandardScaler()',size=18)
plt.legend(['Previsões','Real'],fontsize=12)
plt.show()

# Modelo LinearRegression() com dados normalizados PolynomialFeatures()
poly = PolynomialFeatures()
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Inicializa a regressão linear
rLinear_poly = LinearRegression()
# Ajusta o modelo aos dados de treinamento (aprende os coeficientes)
rLinear_poly.fit(X_train_poly,y_train)
# Faz previsões nos dados de teste
y_pred_poly = rLinear_poly.predict(X_test_poly)


#VISUALIZANDO OS RESULTADOS
#intercept
iRLpoly = (rLinear_poly.intercept_)
print("Intercept: "+str(iRLpoly))
# Visualização dos 03 primeiros resultados
print(f'Predição amostra de 3: {y_pred_poly[:3]}')
#Mean absolute error
maeRLpoly = mean_absolute_error(y_test, y_pred_poly)
print('Erro absoluto médio: %.2f' % maeRLpoly)
#Mean squared error
mRLpoly = mean_squared_error(y_test, y_pred_poly)
print('Erro quadrado médio: %.2f' % mRLpoly)
# R2
r2RLpoly = r2_score(y_test, y_pred_poly)
print('R2: %.6f' % r2RLpoly)

#VALIDAÇÃO DO MODELO 3: CrossValidation
cv_score = np.sqrt(-cross_val_score(rLinear_poly,X_test_poly,y_pred_poly,cv=5,scoring='neg_mean_squared_error'))
cv1RLpoly = cv_score.mean()
cv2RLpoly = cv_score.std()
print("Média: %.6f" %cv1RLpoly)
print("Desvio Padrão: %.6f" %cv2RLpoly)
print("Shape:",y_pred_poly.shape)
print("Scores: ",cv_score)

y_pred_poly = np.array(y_pred_poly)
y_test = np.array(y_test)

plt.figure(figsize=(15,5))
plt.plot(y_pred_poly[:100], linewidth=1.5, color='r')
plt.plot(y_test[:100], linewidth=1.2,color='b')
plt.title('Valores preditos x  Valores reais: Modelo Regressão Linear com PolynomialFeatures() ',size=18)
plt.legend(['Predições','Real'],fontsize=12)
plt.show()

#Modelo RandomForestRegressor()
forest_reg = RandomForestRegressor(n_estimators=20,random_state=10)
forest_reg.fit(X_train , y_train)
y_pred_forest = forest_reg.predict(X_test)

#intercept
#não possui intercepr
iRF = 0
# Visualização dos 03 primeiros resultados
print(f'Predição amostra de 3: {y_pred_forest[:3]}')
#Mean absolute error
maeRF = mean_absolute_error(y_test, y_pred_forest)
print('Erro absoluto médio (MAE): %.2f' % maeRF)
#Mean squared error
mRF = mean_squared_error(y_test, y_pred_forest)
print('Erro quadrado médio (MSE): %.2f' % mRF)
# R2
r2RF = r2_score(y_test, y_pred_forest)
print('R2: %.6f' % r2RF)

#VALIDAÇÃO DO MODELO 4: CrossValidation
cv_score = np.sqrt(-cross_val_score(forest_reg,X_test,y_pred_forest,cv=5,scoring='neg_mean_squared_error'))
cv1RF = cv_score.mean()
cv2RF = cv_score.std()
print("Média: %.6f" %cv1RF)
print("Desvio Padrão: %.6f" %cv2RF)
print("Shape:",y_pred_forest.shape)
print("Scores: ",cv_score)

plt.figure(figsize=(15,5))
plt.plot(y_pred_forest[:100], linewidth=1.5, color='r')
plt.plot(y_test[:100], linewidth=1.2,color='b')
plt.title('Valores preditos x  Valores reais: Random Forest Regressor',size=18)
plt.legend(['Predições','Real'],fontsize=10)
plt.show()

#COMPARAÇÃO DOS RESULTADOS
#RESULTADOS
models = ['LinearRegression()', 'LR/StandardScaler()', 'LR/PolynomialFeatures()', 'RandomForestRegressor()']
intercept = [iRG, iRL, iRLpoly, iRF]
r2 = [r2RG, r2RL, r2RLpoly,r2RF]
MSE = [mRG, mRL, mRLpoly, mRF]
MAE = [maeRG, maeRL, maeRLpoly, maeRF]
cvMean = [cv1RG, cv1RL, cv1RLpoly, cv1RF ]
cvStd = [cv2RG, cv2RL, cv2RLpoly, cv2RF]

df_comp = pd.DataFrame(list(zip(models,intercept,r2,MSE,MAE,cvMean,cvStd )), columns=['Models', 'intercept', 'R2','MSE', 'MAE', 'cvMean','cvStd'])

from decimal import Decimal
df_comp['intercept'] = df_comp['intercept'].map(lambda x: "{:.2f}".format(x))
df_comp['R2'] = df_comp['R2'].map(lambda x: round(x,4))
df_comp['MSE'] = df_comp['MSE'].map(lambda x: "{:.2f}".format(x))
df_comp['MAE'] = df_comp['MAE'].map(lambda x: round(x,2))
df_comp['cvMean'] = df_comp['cvMean'].map(lambda x: round(x,2))
df_comp['cvStd'] = df_comp['cvStd'].map(lambda x: round(x,2))

print(df_comp)