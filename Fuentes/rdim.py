import pandas as pd
import numpy as np
import os
from utility import *

def pc_svd(X, K):
    # X shape: (features, N)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = U[:, :K].T  # K x features
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

def gradW1(Act, W2, L2_lambda=0.001):
    """Gradiente con regularización L2"""
    H = Act['H']
    X_rec = Act['X_rec']
    X = Act['X']
    W1 = Act['W1']
    error = X_rec - X
    batch_size = X.shape[1]
    dH = (W2.T @ error) * deriva_sigmoid(H)
    gW1 = (dH @ X.T) / batch_size + L2_lambda * W1  
    Cost = np.mean(error ** 2) + 0.5 * L2_lambda * np.sum(W1 ** 2)
    return gW1, Cost

def ae_forward(X, W1, W2):
    H = act_sigmoid(W1 @ X)
    X_rec = W2 @ H
    Act = {'X': X, 'H': H, 'X_rec': X_rec, 'W1': W1}
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

def train_miniBatch(Xe, W1, W2, V, S, param, iter_count):
    numBatch = calcula_number_batch(Xe.shape[1], param['BatchSize'])
    Cost = np.zeros(numBatch)
    for n in range(1, numBatch + 1):
        xe = get_miniBatch(n, Xe, param['BatchSize'])
        W2 = ae_pinv(xe, W1, param['C'])
        Act = ae_forward(xe, W1, W2)
        gW1, Cost[n-1] = gradW1(Act, W2, param.get('L2_lambda', 0.001))
        W1, V, S = updW_adam(W1, V, S, gW1, param['mu'], iter_count * numBatch + n)
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
    patience = 50
    patience_counter = 0
    
    for Iter in range(1, param['MaxIter'] + 1):
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        MSE, W1, V, S, W2 = train_miniBatch(Xe, W1, W2, V, S, param, Iter)
        
        # Early stopping
        if MSE < best_mse:
            best_mse = MSE
            best_W1 = W1.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at iteration {Iter}")
            W1 = best_W1
            break
            
        if Iter % 50 == 0:
            print(f"Iter {Iter}: MSE = {MSE:.6f}")
    
    return W1

def normalize_data(X, mean=None, std=None):
    """Normalización consistente"""
    if mean is None:
        mean = np.mean(X, axis=1, keepdims=True)
    if std is None:
        std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # Leer configuración
    config_path = os.path.join(script_dir, 'config_sae.csv')
    config = pd.read_csv(config_path, header=None).values.flatten()
    
    # Leer datos de entrenamiento
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
        'C': 0.05,
        'L2_lambda': 0.001
    }
    
    print(f"Original data shape: {X.shape}")
    
    # Normalizar datos y guardar estadísticas
    X, X_mean, X_std = normalize_data(X)
    
    Vr = []
    stats = {'means': [X_mean], 'stds': [X_std]}
    
    for i in range(param['NumAE']):
        print(f"\n{'='*50}")
        print(f"Training AE-ELM {i+1}...")
        print(f"{'='*50}")
        
        if i == 0:
            param['K'] = param['K1']
        else:
            param['K'] = param['K2']
        
        V_ae = train_ae(X, param)
        Vr.append(V_ae)
        X = act_sigmoid(V_ae @ X)
        
        # Normalizar después de cada capa
        X, mean, std = normalize_data(X)
        stats['means'].append(mean)
        stats['stds'].append(std)
        
        print(f"Output shape after AE {i+1}: {X.shape}")
    
    print(f"\n{'='*50}")
    print("Applying PC-SVD...")
    print(f"{'='*50}")
    V3 = pc_svd(X, param['K3'])
    X = V3 @ X
    Vr.append(V3)
    
    # Normalización final
    X, mean, std = normalize_data(X)
    stats['means'].append(mean)
    stats['stds'].append(std)
    
    print(f"Final reduced shape: {X.shape}")
    
    # Guardar modelo y estadísticas usando dtype=object
    model_path = os.path.join(script_dir, 'model_rdim.npz')
    np.savez(model_path, 
             V1=Vr[0], V2=Vr[1], V3=Vr[2],
             means=np.array(stats['means'], dtype=object),
             stds=np.array(stats['stds'], dtype=object))
    
    # Guardar datos reducidos
    X_reduced = X.T
    output_path = os.path.join(script_dir, 'Xtrain_reduced.csv')
    pd.DataFrame(X_reduced).to_csv(output_path, index=False, header=False)
    
    print(f"\n{'='*50}")
    print("REDUCTION SUMMARY")
    print(f"{'='*50}")
    print(f"Original dimensions: 41")
    print(f"Final dimensions: {X.shape[0]}")
    print(f"Reduction: {(1 - X.shape[0]/41)*100:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"Reduced data saved to: {output_path}")

if __name__ == '__main__':
    main()
