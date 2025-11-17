import pandas as pd
import numpy as np
import os
from utility import *


def updW_adam(W, V, S, gW, mu, n):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    V = beta1 * V + (1 - beta1) * gW
    S = beta2 * S + (1 - beta2) * (gW ** 2)
    V_hat = V / (1 - beta1 ** n) if beta1 != 1 else V
    S_hat = S / (1 - beta2 ** n) if beta2 != 1 else S
    W = W - mu * V_hat / (np.sqrt(S_hat) + eps)
    return W, V, S


def softmax_grad(Act, xe, ye):
    yhat = Act
    error = yhat - ye
    gW = error @ xe.T
    Cost = -np.sum(ye * np.log(yhat + 1e-9)) / ye.shape[1]
    return gW, Cost


def calcula_Batch(N, BatchSize):
    return int(np.ceil(N / BatchSize))


def get_miniBatch(n, X, Y, BatchSize):
    start_idx = (n - 1) * BatchSize
    end_idx = min(n * BatchSize, X.shape[1])
    return X[:, start_idx:end_idx], Y[:, start_idx:end_idx]


def train_miniBatch(Xe, Ye, W, V, S, mu, BatchSize):
    NumBatch = calcula_Batch(Xe.shape[1], BatchSize)
    Cost = np.zeros(NumBatch)
    for n in range(1, NumBatch + 1):
        xe, ye = get_miniBatch(n, Xe, Ye, BatchSize)
        Act = softmax(W @ xe)
        gW, Cost[n-1] = softmax_grad(Act, xe, ye)
        W, V, S = updW_adam(W, V, S, gW, mu, n)
    CostAvg = np.mean(Cost)
    return CostAvg, W, V, S


def train_softmax(X, Y, param):
    Nprev = X.shape[0]
    Nclass = Y.shape[0]
    W = iniW(Nclass, Nprev)
    V = np.zeros_like(W)
    S = np.zeros_like(W)
    cost_history = []
    
    for Iter in range(1, param['MaxIter'] + 1):
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        Ye = Y[:, idx]
        Cost, W, V, S = train_miniBatch(Xe, Ye, W, V, S, param['mu'], param['BatchSize'])
        cost_history.append(Cost)
        if Iter % 100 == 0:
            print(f"Iter {Iter}: Cost = {Cost:.6f}")
            
    return W, cost_history


def main():
    # Obtener la ruta del directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta a la carpeta DATA
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # --- ¡CORRECCIÓN: USAR RUTAS ABSOLUTAS PARA TODOS LOS ARCHIVOS ---
    # 1. Leer config_softmax.csv desde la misma carpeta del script (Fuentes)
    config = pd.read_csv(os.path.join(script_dir, 'config_softmax.csv'), header=None).values.flatten()
    
    # 2. Leer los datos reducidos desde la misma carpeta del script
    X = pd.read_csv(os.path.join(script_dir, 'Xtrain_reduced.csv'), header=None).values.T
    
    # 3. Cargar las etiquetas de entrenamiento desde DATA
    Y = pd.read_csv(os.path.join(data_path, 'classtrain.csv'), header=None).values.T
    
    param = {
        'MaxIter': int(config[0]),
        'mu': float(config[1]),
        'BatchSize': int(config[2])
    }
    
    print("Training Softmax classifier on REDUCED data...")
    W, Cost = train_softmax(X, Y, param)
    
    # Guardar los resultados en la carpeta del script
    np.save(os.path.join(script_dir, 'W_softmax.npy'), W)
    
    cost_history_df = pd.DataFrame({'Cost': Cost})
    cost_history_df.to_csv(os.path.join(script_dir, 'cost_history.csv'), index=False)
    
    print(f"\nTraining complete. Final Cost: {Cost[-1]:.6f}")
    print("Weights saved to 'W_softmax.npy'")

if __name__ == '__main__':
    main()

