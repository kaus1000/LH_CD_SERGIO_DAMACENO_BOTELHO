import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset de treinamento
df_train = pd.read_csv('cars_train.csv')

# Verificar informações básicas do dataset
df_train.info()


# Calcular estatísticas descritivas das variáveis numéricas
df_train.describe()


# Criar histogramas das variáveis numéricas
numeric_cols = ['num_fotos', 'ano_de_fabricacao', 'ano_modelo', 'hodometro', 'num_portas']
df_train[numeric_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()


# Criar gráficos de barras das variáveis categóricas
categorical_cols = ['marca', 'modelo', 'versao', 'cambio', 'tipo', 'blindado', 'cor', 'tipo_vendedor', 'cidade_vendedor', 'estado_vendedor', 'anunciante', 'entrega_delivery', 'troca', 'elegivel_revisao', 'dono_aceita_troca', 'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago', 'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda', 'veiculo_alienado']
fig, axes = plt.subplots(8, 3, figsize=(18, 24))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df_train, ax=axes[i])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()



# Calcular a contagem das marcas mais populares em cada estado
popular_brands = ['Volkswagen', 'Chevrolet', 'Fiat', 'Ford', 'Renault']
df_popular_brands = df_train[df_train['marca'].isin(popular_brands)]
popular_brands_by_state = df_popular_brands.groupby(['estado_vendedor', 'marca']).size().unstack()

# Plotar gráfico de barras empilhadas
popular_brands_by_state.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.xlabel('Estado')
plt.ylabel('Contagem')
plt.title('Distribuição das Marcas Populares por Estado')
plt.legend(title='Marca')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Filtrar picapes com transmissão automática
df_pickup_auto = df_train[(df_train['tipo'] == 'picape') & (df_train['cambio'] == 'automático')]

# Calcular a contagem das picapes com transmissão automática em cada estado
pickup_auto_by_state = df_pickup_auto['estado_vendedor'].value_counts()

# Plotar gráfico de barras
pickup_auto_by_state.plot(kind='bar', figsize=(12, 8))
plt.xlabel('Estado')
plt.ylabel('Contagem')
plt.title('Distribuição de Picapes com Transmissão Automática por Estado')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Calcular a média dos preços de revenda para cada cor de veículo
mean_price_by_color = df_train.groupby('cor')['preco'].mean().sort_values()

# Plotar gráfico de barras
mean_price_by_color.plot(kind='bar', figsize=(12, 8))
plt.xlabel('Cor')
plt.ylabel('Preço Médio')
plt.title('Preço Médio de Revenda por Cor do Veículo')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Plotar gráfico de dispersão
plt.figure(figsize=(12, 8))
plt.scatter(df_train['num_fotos'], df_train['preco'])
plt.xlabel('Quantidade de Fotos')
plt.ylabel('Preço')
plt.title('Relação entre Quantidade de Fotos e Preço de Revenda')
plt.tight_layout()
plt.show()


# Plotar gráfico de dispersão
plt.figure(figsize=(12, 8))
plt.scatter(df_train['hodometro'], df_train['preco'])
plt.xlabel('Hodômetro')
plt.ylabel('Preço')
plt.title('Relação entre Hodômetro e Preço de Revenda')
plt.tight_layout()
plt.show()

