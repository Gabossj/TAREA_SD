import pandas as pd
import numpy as np
import os
from utility import *

def pc_svd(X, K):
    mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean
    N = X.shape[1]
    Y = X_centered / np.sqrt(N - 1)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    V = U[:, :K].T
    return V, mean

def updW_adam(W, V, S, gW, mu, t):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    V = beta1 * V + (1 - beta1) * gW
    S = beta2 * S + (1 - beta2) * (gW ** 2)
    factor = np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    W = W - mu * factor * V / (np.sqrt(S) + eps)
    return W, V, S

def gradW1(Act, W2):
    H = Act['H']
    X_rec = Act['X_rec']
    X = Act['X']
    batch_size = X.shape[1]
    error = X_rec - X
    dH = (W2.T @ error) * deriva_sigmoid(H)
    gW1 = (dH @ X.T) / batch_size
    Cost = np.mean(error ** 2)
    return gW1, Cost

def ae_forward(X, W1, W2): 
    H = act_sigmoid(W1 @ X)
    X_rec = W2 @ H
    Act = {'X': X, 'H': H, 'X_rec': X_rec, 'W1': W1}
    return Act 

def ae_pinv(X, W1, C):
    H = act_sigmoid(W1 @ X)
    HtH = H @ H.T
    K = H.shape[0]
    lam = 1.0 / C
    W2 = X @ H.T @ np.linalg.inv(HtH + lam * np.eye(K))
    return W2 

def calcula_number_batch(N, BatchSize):
    return int(np.ceil(N / BatchSize))

def get_miniBatch(n, X, BatchSize):
    start_idx = (n - 1) * BatchSize
    end_idx = min(n * BatchSize, X.shape[1])
    return X[:, start_idx:end_idx]

def train_miniBatch(Xe, W1, W2, V, S, param, iter_count):
    numBatch = calcula_number_batch(Xe.shape[1], param['BatchSize'])
    Cost = np.zeros(numBatch)
    
    for n in range(1, numBatch + 1):
        xe = get_miniBatch(n, Xe, param['BatchSize'])
        W2 = ae_pinv(xe, W1, param['C'])
        Act = ae_forward(xe, W1, W2)
        gW1, Cost[n-1] = gradW1(Act, W2)
        t = iter_count * numBatch + n
        W1, V, S = updW_adam(W1, V, S, gW1, param['mu'], t)
    
    MSEavg = np.mean(Cost)
    return MSEavg, W1, V, S, W2

def train_ae(X, param):
    Nprev = X.shape[0]
    Nnext = param['K']
    W1 = iniW(Nnext, Nprev)
    W2 = iniW(Nprev, Nnext)
    V = np.zeros_like(W1)
    S = np.zeros_like(W1)
    best_mse = float('inf')
    best_W1 = W1.copy()
    
    print(f"Training AE: {Nprev} -> {Nnext}")
    
    for Iter in range(1, param['MaxIter'] + 1):
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        MSE, W1, V, S, W2 = train_miniBatch(Xe, W1, W2, V, S, param, Iter)
        
        if MSE < best_mse:
            best_mse = MSE
            best_W1 = W1.copy()
        
        if Iter % 50 == 0:
            print(f"Iter {Iter}/{param['MaxIter']}: MSE = {MSE:.6f}")
    
    print(f"Best MSE: {best_mse:.6f}\n")
    return best_W1

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    config_path = os.path.join(script_dir, 'config_sae.csv')
    config = pd.read_csv(config_path, header=None).values.flatten()
    
    dtrain_path = os.path.join(data_path, 'dtrain.csv')
    X = pd.read_csv(dtrain_path, header=None).values.T
    
    param = {
        'K1': int(config[0]),
        'K2': int(config[1]),
        'K3': int(config[2]),
        'MaxIter': int(config[3]),
        'mu': float(config[4]),
        'NumAE': 2,
        'BatchSize': 128,
        'C': 10
    }
    
    print(f"Config: K1={param['K1']}, K2={param['K2']}, K3={param['K3']}, MaxIter={param['MaxIter']}, mu={param['mu']}, BatchSize={param['BatchSize']}, C={param['C']}\n")
    
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    Vr = []
    svd_means = []
    
    param['K'] = param['K1']
    V1 = train_ae(X, param)
    Vr.append(V1)
    X = act_sigmoid(V1 @ X)
    
    param['K'] = param['K2']
    V2 = train_ae(X, param)
    Vr.append(V2)
    X = act_sigmoid(V2 @ X)
    
    V3, svd_mean = pc_svd(X, param['K3'])
    X = V3 @ (X - svd_mean)
    Vr.append(V3)
    svd_means.append(svd_mean)
    
    
    reduction_pct = (1 - X.shape[0] / 41) * 100
    print(f"Dimensions: 41 -> {X.shape[0]} (Reduction: {reduction_pct:.2f}%)")
    
    model_path = os.path.join(script_dir, 'model_rdim.npz')
    np.savez(model_path, V1=Vr[0], V2=Vr[1], V3=Vr[2], X_mean=X_mean, X_std=X_std, svd_mean=svd_means[0])
    
    X_reduced = X.T
    output_path = os.path.join(script_dir, 'Xtrain_reduced.csv')
    pd.DataFrame(X_reduced).to_csv(output_path, index=False, header=False)

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nExecution time: {elapsed:.2f}s")
    print("✓ F1 MET" if elapsed <= 45 else "✗ F1 NOT MET")