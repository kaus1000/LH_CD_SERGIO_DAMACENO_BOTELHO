# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Manipulação de dados
import pandas as pd
import numpy as np

#LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Normalização dos dados
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


def load_training_Data():
    print("Carregando o dataset..\n")
    df = pd.read_csv('cars_train.csv', encoding='utf16', delimiter='\t')
    return df

def load_Test_Data():
    print("Carregando o dataset de teste..\n")
    df_test = pd.read_csv('cars_test.csv', encoding='utf16', delimiter='\t')
    return df_test

# Função para realizar a análise exploratória dos dados
def exploratory_Data_Analysis(df):
    print("Realizando a análise exploratória dos dados:\n")
    print("Total de linhas e colunas:")
    print(df.shape)
    print("Primeiras linhas do dataset:")
    print(df.head())
    print("Amostra aleatória do dataset:")
    print(df.sample(5))
    print("Informações sobre o dataset:")
    print(df.info())
    
def predict_test_data(model):
    # Carregar os dados de teste
    print("Carregando os dados de teste...")
    df_test = load_Test_Data()

    # Verificar a existência de valores ausentes nos dados
    print("Verificando a existência de valores ausentes...")
    df_test = missing_Data_Verification(df_test)

    # Transformar as colunas categóricas
    print("Transformando as colunas categóricas...")
    df_test = transforming_categorical_columns(df_test)

    # Lidar com valores ausentes nos dados
    print("Lidando com valores ausentes...")
    df_test = missing_Values(df_test)

    # Agora podemos fazer previsões com o modelo
    print("Fazendo previsões...")
    y_pred = model.predict(df_test)

    # Criar um novo DataFrame com os IDs e os preços previstos
    print("Criando o DataFrame de previsões...")
    predictions = pd.DataFrame({'id': df_test['id'], 'preco': y_pred})

    # Salvar as previsões em um arquivo CSV
    print("Salvando as previsões em um arquivo CSV...")
    predictions.to_csv('predictions.csv', index=False)

    print("Previsões concluídas!")
    

def transforming_categorical_columns(df):
    #TRANSFORMANDO CATEGORIAS EM VALORES
    #Ajuste das variáveis categóricas para valores 0 e 1
    print("Transformando colunas categoricas em valores 0 e 1.\n")
    categorical_columns = ['id', 'marca', 'modelo', 'versao', 'cambio', 'tipo', 
                        'blindado', 'cor', 'tipo_vendedor', 'cidade_vendedor', 
                        'estado_vendedor', 'anunciante', 'dono_aceita_troca',
                        'veiculo_único_dono', 'revisoes_concessionaria', 'ipva_pago',
                        'veiculo_licenciado', 'garantia_de_fábrica', 'revisoes_dentro_agenda']
    # Criar uma cópia do DataFrame original
    df_encoded = df.copy()
    # Aplicar Label Encoding em todas as colunas categóricas
    label_encoder = LabelEncoder()
    try:
        for col in categorical_columns:
            df_encoded[col] = label_encoder.fit_transform(df[col])
    except:
        pass
        
    return df_encoded

def visualize_data(df):
    # Visualizando os dados
    print("Carregando graficos para vizualização dos dados.\n")

    print(df.shape)
    df.hist(bins=80, figsize=(16,12),color="red")
    plt.show()
    fig, axs = plt.subplots(3, 2, figsize=(8, 13))

    axs[0, 0].hist(df['preco'], bins=30, color='skyblue', edgecolor='black')
    axs[0, 0].set_title('Distribuição do Preço', fontsize=10)
    axs[0, 0].set_xlabel('Preço',fontsize=8)
    axs[0, 0].set_ylabel('Frequência')
    axs[0, 0].grid(axis='y', alpha=0.75)

    axs[0, 1].hist(df['ano_de_fabricacao'], bins=30, color='skyblue', edgecolor='black')
    axs[0, 1].set_title('Distribuição do Ano de Fabricação' , fontsize=10)
    axs[0, 1].set_xlabel('Ano de Fabricação' , fontsize=8)
    axs[0, 1].set_ylabel('Frequência' , fontsize=8)
    axs[0, 1].grid(axis='y', alpha=0.75)

    top_states = df['estado_vendedor'].value_counts().nlargest(10)
    top_states.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[1, 0])
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45,fontsize=6)
    axs[1, 0].set_title('10 Estados com mais Registros' , fontsize=8)
    axs[1, 0].set_ylabel('Frequência' , fontsize=6)
    axs[1, 0].grid(axis='x', alpha=0.75)

    transmission_counts = df['cambio'].value_counts()
    transmission_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[1, 1])
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45, fontsize=6)
    axs[1, 1].set_title('Tipos de Transmissão' , fontsize=8)
    axs[1, 1].set_ylabel('Frequência', fontsize=6)
    axs[1, 1].grid(axis='y', alpha=0.75)

    top_brands = df['marca'].value_counts().nlargest(10)
    top_brands.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[2, 0])
    axs[2, 0].set_xticklabels(axs[2, 0].get_xticklabels(), rotation=45, fontsize=6)
    axs[2, 0].set_ylabel('Frequência', fontsize=8)
    axs[2, 0].grid(axis='y', alpha=0.75)

    mean_price_by_brand = df.groupby('marca')['preco'].mean()
    sns.barplot(x=mean_price_by_brand.index, y=mean_price_by_brand.values, ax=axs[2, 1])
    axs[2, 1].set_xticklabels(axs[2, 1].get_xticklabels(), rotation=75, fontsize=7)
    axs[2, 1].set_xlabel('Marca', fontsize=8)
    axs[2, 1].set_ylabel('Preço Médio', fontsize=6)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

def missing_Data_Verification(df):
    #VERIFICAÇÃO DE DADOS FALTANTES
    print("Verificando dados faltantes..\n")
    print('TOTAL Null\n')
    print(df.isnull().sum(),'\n')
    print('TOTAL NA\n')
    print(df.isna().sum(),)
    #Retirada da coluna vazia veiculo_alienado e elegivel_revisao
    columns_to_drop = ['veiculo_alienado','elegivel_revisao']
    df = df.drop(columns=columns_to_drop)
    
    return df

def correlation_features(df):
    #CORRELAÇÃO
    plt.figure(figsize=(12,12))
    sns.set_theme()
    sns.heatmap(df.corr(), annot=True, fmt=".1f",linewidth=.5 ,cmap="RdBu")
    plt.show()
    correlation = df.corr()["preco"].sort_values(ascending = False)
    print("Correlação entre as variáveis e a váriavel alvo:\n")
    print(correlation.head(10))  

def verification_hypotheses(df):
    
    #VERIFICAÇÃO DE HIPÓTESES
    print("Checando hipóteses..\n")
    popular_brands = ['VOLKSWAGEN', 'CHEVROLET', 'FORD']
    avg_price_popular_brands = df[df['marca'].isin(popular_brands)]['preco'].mean()
    avg_price_other_brands = df[~df['marca'].isin(popular_brands)]['preco'].mean()

    print("Hipótese 1: Carros de marcas populares são mais baratos do que os de outras marcas")
    print(f"Preço médio dos carros de marcas populares: R$ {avg_price_popular_brands:.2f}")
    print(f"Preço médio dos carros de outras marcas: R$ {avg_price_other_brands:.2f}\n")

    avg_price_auto = df[df['cambio'] == 'Automática']['preco'].mean()
    avg_price_other_trans = df[df['cambio'] != 'Automática']['preco'].mean()

    print("Hipótese 2: Carros com transmissão automática são mais caros do que carros com outros tipos de transmissão")
    print(f"Preço médio dos carros com transmissão automática: R$ {avg_price_auto:.2f}")
    print(f"Preço médio dos carros com outros tipos de transmissão: R$ {avg_price_other_trans:.2f}\n")


    avg_price_warranty = df[df['garantia_de_fábrica'] == 'Garantia de fábrica']['preco'].mean()
    avg_price_no_warranty = df[df['garantia_de_fábrica'] != 'Garantia de fábrica']['preco'].mean()

    print("Hipótese 3: Carros que ainda estão na garantia de fábrica são mais caros do que aqueles que não estão")
    print(f"Preço médio dos carros na garantia de fábrica: R$ {avg_price_warranty:.2f}")
    print(f"Preço médio dos carros sem garantia de fábrica: R$ {avg_price_no_warranty:.2f}\n")

    df_popular_brands = df[df['marca'].isin(popular_brands)]
    avg_price_state_popular_brands = df_popular_brands.groupby('estado_vendedor')['preco'].mean()
    best_state_to_sell_popular_brand = avg_price_state_popular_brands.idxmax()

    print("Pergunta de negócios 1: Qual é o melhor estado registrado na base de dados para vender um carro de marca popular e por quê?")
    print(f"Melhor estado para vender carro de marca popular: {best_state_to_sell_popular_brand}\n")

    df_automatic_pickups = df[(df['cambio'] == 'Automática') & (df['tipo'] == 'Picape')]
    avg_price_state_automatic_pickups = df_automatic_pickups.groupby('estado_vendedor')['preco'].mean()
    best_state_to_buy_automatic_pickup = avg_price_state_automatic_pickups.idxmin()

    print("Pergunta de negócios 2: Qual é o melhor estado para comprar uma picape com transmissão automática e por quê?")
    print(f"Melhor estado para comprar picape automática: {best_state_to_buy_automatic_pickup}\n")

    df_warranty = df[df['garantia_de_fábrica'] == 'Garantia de fábrica']
    avg_price_state_warranty = df_warranty.groupby('estado_vendedor')['preco'].mean()
    best_state_to_buy_warranty = avg_price_state_warranty.idxmin()

    print("Pergunta de negócios 3: Qual é o melhor estado para comprar carros que ainda estão na garantia de fábrica e por quê?")
    print(f"Melhor estado para comprar carro com garantia de fábrica: {best_state_to_buy_warranty}\n")

def missing_Values(df):
    print("tratando valores ausentes.\n")
    df.dropna(inplace=True)
    
    return df

def separation_Data_Training_Test(df):
    #Separação dos dados em Treino e Teste
    #DADOS DAS VARIÁVEIS INDEPENDENTES
    X = df.drop(["preco"], axis = 1)
    #DADOS DA VARIÁVEL DEPENDENTE, QUE QUEREMOS ENCONTRAR
    y = df["preco"]
    
    return X,y

def division_the_dataframe_test_training(X,y):
    #DIVISÃO DO DATA FRAME EM TEST E TREINAMENTO 80% TREINAMENTO E 20% TESTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)
    #Tamanho das Amostras de Treino e Teste
    print(f"X_train dimensões {X_train.shape} e tamanho {X_train.size}")
    print(f"X_test dimensões {X_test.shape} e tamanho {X_test.size}")
    print(f"y_train dimensões {y_train.shape} e tamanho {y_train.size}")
    print(f"y_test dimensões {y_test.shape} e tamanho {y_test.size}")
    
    return X_train, y_train, X_test ,y_test

def calculate_metrics(y_test, y_pred):
    result = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
    }

    return result

def print_results(results):
    model_names = ['Modelo Regressão Linear', 'Modelo Regressão Linear com StandardScaler', 'Modelo Regressão Linear com PolynomialFeatures', 'Modelo Regressão de Floresta Aleatória']
    for i, result in enumerate(results):
        print(f'Modelo {model_names[i]}:')
        print('MAE:', result['MAE'])
        print('MSE:', result['MSE'])
        print('R2:', result['R2'])
        print('----------')

def print_cross_val_score_results(model, X_test, y_pred, model_name):
    cv_score = np.sqrt(-cross_val_score(model, X_test, y_pred, cv=5, scoring='neg_mean_squared_error'))
    cv_mean = cv_score.mean()
    cv_std = cv_score.std()
    print("Modelo: {}".format(model_name))
    print("-------" * 10)
    print("Média do erro quadrático médio (RMSE) na validação cruzada (Cross-Validation): {:.6f}".format(cv_mean))
    print("Desvio padrão do erro quadrático médio (RMSE) na validação cruzada (Cross-Validation): {:.6f}".format(cv_std))
    print("Shape da predição:", y_pred.shape)
    print("Scores do erro quadrático médio negativo (neg_mean_squared_error) na validação cruzada (Cross-Validation): ", cv_score)
    print("\n")

def train_linear_regression(X_train, y_train, X_test, y_test):
    # Criando o modelo LinearRegression()
    regLinear = LinearRegression()
    # Realizar treinamento do modelo
    regLinear.fit(X_train, y_train)
    # Realizar predição com os dados separados para teste
    pred_regLinear = regLinear.predict(X_test)

    # Cálculo das métricas
    result = calculate_metrics(y_test, pred_regLinear)

    # Preparar dados para plotagem
    plot_data = (pred_regLinear, y_test, 'Valores preditos x  Valores reais : Modelo Regressão Linear')

    return pred_regLinear,regLinear,result, plot_data

def train_linear_regression_with_scaler(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    rLinear = LinearRegression()
    rLinear.fit(X_train_scaler,y_train)
    y_pred_scaler = rLinear.predict(X_test_scaler)

    result = calculate_metrics(y_test, y_pred_scaler)

    plot_data = (y_pred_scaler, y_test, 'Valores previstos x Valores reais: Modelo Regressão Linear com StandardScaler()')

    return X_test_scaler, y_pred_scaler,rLinear,result, plot_data

def train_linear_regression_with_poly(X_train, y_train, X_test, y_test):
    poly = PolynomialFeatures()
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    rLinear_poly = LinearRegression()
    rLinear_poly.fit(X_train_poly,y_train)
    y_pred_poly = rLinear_poly.predict(X_test_poly)

    result = calculate_metrics(y_test, y_pred_poly)

    plot_data = (y_pred_poly, y_test, 'Valores preditos x  Valores reais: Modelo Regressão Linear com PolynomialFeatures() ')

    return X_test_poly, y_pred_poly,rLinear_poly,result, plot_data

def train_random_forest(X_train, y_train, X_test, y_test):
    print("Treinando modelo de random forest...")
    forest_reg = RandomForestRegressor(max_depth= None, min_samples_split= 10, n_estimators= 300,random_state=10)
    forest_reg.fit(X_train , y_train)
    y_pred_forest = forest_reg.predict(X_test)

    result = calculate_metrics(y_test, y_pred_forest)

    plot_data = (y_pred_forest, y_test, 'Valores preditos x  Valores reais: Random Forest Regressor')

    return y_pred_forest,forest_reg, result, plot_data


def plot_results(plot_data):
    for plot in plot_data:
        y_pred, y_test, title = plot
        y_test = np.array(y_test)
        plt.figure(figsize=(15, 5))
        plt.plot(y_pred[:100], linewidth=1.5, color='r')
        plt.plot(y_test[:100], linewidth=1.2, color='b')
        plt.title(title, size=18)
        plt.legend(['Predições', 'Real'], fontsize=12)
        plt.show()

def training_the_models(X_train, y_train, X_test ,y_test):
    # Armazenar resultados para comparação
    results = []

    # Armazenar plots para visualização
    plots = []

    # Modelo LinearRegression()
    pred_regLinear,regLinear,result, plot = train_linear_regression(X_train, y_train, X_test, y_test)
    results.append(result)
    plots.append(plot)
    print_cross_val_score_results(regLinear, X_test, pred_regLinear, "Regressão Linear")

    # Modelo LinearRegression() com dados normalizados StandardScaler()
    X_test_scaler, y_pred_scaler,rLinear, result, plot = train_linear_regression_with_scaler(X_train, y_train, X_test, y_test)
    results.append(result)
    plots.append(plot)
    print_cross_val_score_results(rLinear, X_test_scaler, y_pred_scaler, "LinearRegression() com dados normalizados StandardScaler")

    # Modelo LinearRegression() com dados normalizados PolynomialFeatures()
    X_test_poly, y_pred_poly, rLinear_poly,result, plot = train_linear_regression_with_poly(X_train, y_train, X_test, y_test)
    results.append(result)
    plots.append(plot)
    print_cross_val_score_results(rLinear_poly, X_test_poly, y_pred_poly, "Modelo LinearRegression() com dados normalizados PolynomialFeatures")

    # Modelo RandomForestRegressor()
    y_pred_forest, forest_reg,result, plot = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(result)
    plots.append(plot)
    print_cross_val_score_results(forest_reg, X_test, y_pred_forest, "Modelo RandomForestRegressor")


    return forest_reg, results, plots


def main():
    # Carregar os dados
    df = load_training_Data()
    
    # Análise Exploratória dos Dados
    exploratory_Data_Analysis(df)
    
    # Verificação de dados faltantes
    df= missing_Data_Verification(df)
    
    # Visualização dos dados
    visualize_data(df)
    # Verificação de hipóteses
    verification_hypotheses(df)
    
    # Transformando as colunas categóricas em números
    df = transforming_categorical_columns(df)
    
    #plota as correlações ao preço
    correlation_features(df)
    
    # Tratamento de valores ausentes
    df=missing_Values(df)
    
    # Separação dos dados em treino e teste
    X, y = separation_Data_Training_Test(df)
    
    # Divisão dos dados em treinamento e teste
    X_train, y_train, X_test, y_test = division_the_dataframe_test_training(X, y)
    
    # Treinamento dos modelos e exibição dos resultados
    forest_reg,results,plot_data  = training_the_models(X_train, y_train, X_test, y_test)
    
    # Exibir resultados dos dados de treinamento
    print_results(results)
    
    plot_results(plot_data)
    # Prever os preços para os dados de teste usando o modelo treinado
    predict_test_data(forest_reg)

if __name__ == "__main__":
    main()
