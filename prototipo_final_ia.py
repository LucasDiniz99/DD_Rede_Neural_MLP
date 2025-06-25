import pandas as pd
from sklearn.neural_network import MLPRegressor
import os
import sys
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np

# Ocultando warnings já tratados
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Limpar console
def clearConsole():
    os.system('cls')

# Carregar dados do csv
def carregarDados():
    data = pd.read_csv("dd.csv")
    X = data.iloc[:,0:9]
    y = data.iloc[:,9:10]
    return X, y

# Gera modelo e métricas do modelo
def prepararModelo():
    X, y = carregarDados()    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)  
    X_test = X_scaler.transform(X_test)  

    mlp = MLPRegressor(
        hidden_layer_sizes=(80, 20, ),  # Arquitetura da rede: 2 camadas ocultas
        activation='relu',            # Função de ativação
        solver='adam',                # Otimizador
        alpha=0.001,                  # Termo de regularização
        batch_size=16,                # Tamanho do batch
        learning_rate='adaptive',     # Taxa de aprendizado adaptativa
        max_iter=250,                # Número máximo de iterações
        early_stopping=True,          # Parada antecipada
        validation_fraction=0.1,      # Fração para validação
        random_state=42,
        verbose=False
    )
    
    # Execução do treino
    mlp.fit(X_train, y_train.values.ravel())
    # The final step is to make predictions on our test data. To do so, execute the following script:
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mlp, mse, r2, X_scaler

# Chama um modelo para previsão
def predict(model, X_scaler, user_input):
    input_array = np.array(user_input).reshape(1, -1)
    X_test = X_scaler.transform(input_array)
    y_pred = model.predict(X_test)
    return y_pred    

# 3. Interface para entrada manual
def get_user_input():
    print("\nPor favor, insira os valores para cada coluna:")
    
    columns = [
        "Tempo de execucao",
        "Alcance",
        "Alvo",
        "Efeito",
        "Duracao",
        "Resistencia",
        "Custos extra",
        "Circulo",
        "Dado"
    ]
    
    user_data = []
    for i, col in enumerate(columns, 1):
        while True:
            try:
                value = float(input(f"[{i}/9] {col}: "))
                user_data.append(value)
                break
            except ValueError:
                print("Por favor, insira um número válido.")
    
    return np.array([user_data])

# Selecionando melhor modelo
print("=======================\nIniciando seleção de modelo:\n")
melhorModelo, melhorMse, melhorR2, melhorX_scaler = prepararModelo()
for i in range(0, 20):
    newModelo, newMse, newR2, melhorX_scaler = prepararModelo()
    print(f"Métricas {i}:: [MSE] = {newMse:.4f}; [R²] = {newR2:.4f}\n")
    if(newMse < melhorMse):
        melhorModelo = newModelo
        melhorMse = newMse
        melhorR2 = newR2

print("\n=======================\nMelhor modelo:")
print(f"[MSE]: {melhorMse:.4f}\n[R²]: {melhorR2:.4f}\n")
    
print("\n[Sistema pronto para uso]\n")
    
# Interface do usuário
while True:
    print("\n" + "="*50)
    print("MENU PRINCIPAL")
    print("1. Fazer nova previsão")
    print("2. Sair")
    
    choice = input("Escolha uma opção (1/2): ")
    
    if choice == '1':
        try:
            # Obter entrada do usuário
            user_input = get_user_input()
            
            # Fazer previsão
            prediction = predict(melhorModelo, melhorX_scaler, user_input)[0]
                        
            print("Aferindo resultados:\n")
            print(f"Belefíncios previstos para a magia: {prediction:.4f}\n")
            
            if(prediction > 36):
                print('A magia descrita é desbalanceada, seu benefício previsto é muito maior que os custos')
            elif(prediction <= 36 and prediction >= 0):
                print('A magia descrita tem valores balanceados')
            else:
                print('A magia descrita é desbalanceada, seus custos são muito superiores aos benefícios previstos')
            
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            
    elif choice == '2':
        sys.exit(0)        
    else:
        clearConsole()
