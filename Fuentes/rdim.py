import pandas as pd
import numpy as np
import os
from utility import *

def pc_svd(X, K):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = U[:, :K].T
    return V

def updW_adam(W, V, S, gW, mu, t):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    V = beta1 * V + (1 - beta1) * gW
    S = beta2 * S + (1 - beta2) * (gW ** 2)
    V_hat = V / (1 - beta1 ** t)
    S_hat = S / (1 - beta2 ** t)
    W = W - mu * V_hat / (np.sqrt(S_hat) + eps)
    return W, V, S

def gradW1(Act, W2):
    H = Act['H']
    X_rec = Act['X_rec']
    X = Act['X']
    error = X_rec - X
    batch_size = X.shape[1]
    dH = (W2.T @ error) * deriva_sigmoid(H)
    gW1 = (dH @ X.T) / batch_size
    Cost = np.mean(error ** 2)
    return gW1, Cost

def ae_forward(X, W1, W2):
    H = act_sigmoid(W1 @ X)
    X_rec = W2 @ H
    Act = {'X': X, 'H': H, 'X_rec': X_rec}
    return Act

def ae_pinv(X, W1, C):
    H = act_sigmoid(W1 @ X)
    HtH = H @ H.T
    W2 = X @ H.T @ np.linalg.inv(HtH + C * np.eye(H.shape[0]))
    return W2

def calcula_number_batch(N, BatchSize):
    return int(np.ceil(N / BatchSize))

def get_miniBatch(n, X, BatchSize):
    start_idx = (n - 1) * BatchSize
    end_idx = min(n * BatchSize, X.shape[1])
    return X[:, start_idx:end_idx]

def train_miniBatch(Xe, W1, W2, V, S, param):
    numBatch = calcula_number_batch(Xe.shape[1], param['BatchSize'])
    Cost = np.zeros(numBatch)
    for n in range(1, numBatch + 1):
        xe = get_miniBatch(n, Xe, param['BatchSize'])
        W2 = ae_pinv(xe, W1, param['C'])
        Act = ae_forward(xe, W1, W2)
        gW1, Cost[n-1] = gradW1(Act, W2)
        W1, V, S = updW_adam(W1, V, S, gW1, param['mu'], n)
    MSEavg = np.mean(Cost)
    return MSEavg, W1, V, S, W2

def train_ae(X, param):
    Nprev = X.shape[0]
    Nnext = param['K']
    W1 = iniW(Nnext, Nprev)
    W2 = iniW(Nprev, Nnext)
    V = np.zeros_like(W1)
    S = np.zeros_like(W1)
    for Iter in range(1, param['MaxIter'] + 1):
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        MSE, W1, V, S, W2 = train_miniBatch(Xe, W1, W2, V, S, param)
        if Iter % 50 == 0:
            print(f"Iter {Iter}: MSE = {MSE:.6f}")
    return W1

def main():
    # Obtener la ruta del directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta a la carpeta DATA (un nivel arriba y luego a DATA)
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # Leer config_sae.csv desde la carpeta Fuentes
    config_path = os.path.join(script_dir, 'config_sae.csv')
    config = pd.read_csv(config_path, header=None).values.flatten()
    
    # Leer dtrain.csv desde la carpeta DATA
    dtrain_path = os.path.join(data_path, 'dtrain.csv')
    X = pd.read_csv(dtrain_path, header=None).values.T
    
    param = {
        'K1': int(config[0]),
        'K2': int(config[1]),
        'K3': int(config[2]),
        'MaxIter': int(config[3]),
        'mu': float(config[4]),
        'NumAE': 2,
        'BatchSize': 256,
        'C': 0.1
    }
    
    # Normalizar datos
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    
    Vr = []
    for i in range(param['NumAE']):
        print(f"\nTraining AE-ELM {i+1}...")
        if i == 0:
            param['K'] = param['K1']
        else:
            param['K'] = param['K2']
        V_ae = train_ae(X, param)
        Vr.append(V_ae)
        X = act_sigmoid(V_ae @ X)
    
    print("\nApplying PC-SVD...")
    V3 = pc_svd(X, param['K3'])
    X = V3 @ X
    Vr.append(V3)
    
    # Guardar modelo en la carpeta Fuentes
    model_path = os.path.join(script_dir, 'model_rdim.npz')
    np.savez(model_path, V1=Vr[0], V2=Vr[1], V3=Vr[2])
    
    # Guardar datos reducidos en la carpeta Fuentes
    X_reduced = X.T
    output_path = os.path.join(script_dir, 'Xtrain_reduced.csv')
    pd.DataFrame(X_reduced).to_csv(output_path, index=False, header=False)
    
    print(f"\nReduction complete: {41} -> {X.shape[0]} dimensions")
    print(f"Reduction: {(1 - X.shape[0]/41)*100:.2f}%")
    print(f"Reduced data saved to: {output_path}")

if __name__ == '__main__':
    main()