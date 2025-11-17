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


def softmax_grad(Act, xe, ye, W, L2_lambda=0.0001):
    """Gradiente con regularización L2"""
    yhat = Act
    error = yhat - ye
    batch_size = ye.shape[1]
    gW = (error @ xe.T) / batch_size + L2_lambda * W
    Cost = -np.sum(ye * np.log(yhat + 1e-9)) / batch_size
    return gW, Cost


def calcula_Batch(N, BatchSize):
    return int(np.ceil(N / BatchSize))


def get_miniBatch(n, X, Y, BatchSize):
    start_idx = (n - 1) * BatchSize
    end_idx = min(n * BatchSize, X.shape[1])
    return X[:, start_idx:end_idx], Y[:, start_idx:end_idx]


def train_miniBatch(Xe, Ye, W, V, S, mu, BatchSize, L2_lambda, iter_num):
    NumBatch = calcula_Batch(Xe.shape[1], BatchSize)
    Cost = np.zeros(NumBatch)
    for n in range(1, NumBatch + 1):
        xe, ye = get_miniBatch(n, Xe, Ye, BatchSize)
        Act = softmax(W @ xe)
        gW, Cost[n-1] = softmax_grad(Act, xe, ye, W, L2_lambda)
        W, V, S = updW_adam(W, V, S, gW, mu, iter_num * NumBatch + n)
    CostAvg = np.mean(Cost)
    return CostAvg, W, V, S


def evaluate_accuracy(X, Y, W):
    """Calcular accuracy en conjunto de datos"""
    predictions = softmax(W @ X)
    y_pred = np.argmax(predictions, axis=0)
    y_true = np.argmax(Y, axis=0)
    accuracy = np.mean(y_pred == y_true)
    return accuracy


def train_softmax(X, Y, param):
    Nprev = X.shape[0]
    Nclass = Y.shape[0]
    W = iniW(Nclass, Nprev) * 0.1  # Inicialización más pequeña
    V = np.zeros_like(W)
    S = np.zeros_like(W)
    cost_history = []
    acc_history = []
    
    best_acc = 0
    best_W = W.copy()
    patience = 100
    patience_counter = 0
    
    # Learning rate scheduling
    initial_mu = param['mu']
    
    for Iter in range(1, param['MaxIter'] + 1):
        # Decay del learning rate
        if Iter > 500:
            param['mu'] = initial_mu * 0.5
        if Iter > 1000:
            param['mu'] = initial_mu * 0.1
        if Iter > 1500:
            param['mu'] = initial_mu * 0.05
            
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        Ye = Y[:, idx]
        
        Cost, W, V, S = train_miniBatch(
            Xe, Ye, W, V, S, 
            param['mu'], 
            param['BatchSize'],
            param.get('L2_lambda', 0.0001),
            Iter
        )
        cost_history.append(Cost)
        
        # Evaluar accuracy cada 50 iteraciones
        if Iter % 50 == 0:
            acc = evaluate_accuracy(X, Y, W)
            acc_history.append(acc)
            print(f"Iter {Iter}: Cost = {Cost:.6f}, Accuracy = {acc:.4f}")
            
            # Early stopping basado en accuracy
            if acc > best_acc:
                best_acc = acc
                best_W = W.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience // 50:  # Dividir por 50 porque evaluamos cada 50 iters
                print(f"Early stopping at iteration {Iter}. Best accuracy: {best_acc:.4f}")
                W = best_W
                break
    
    # Usar el mejor modelo
    W = best_W
    final_acc = evaluate_accuracy(X, Y, W)
    print(f"\nFinal Training Accuracy: {final_acc:.4f}")
            
    return W, cost_history


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # Leer configuración
    config = pd.read_csv(os.path.join(script_dir, 'config_softmax.csv'), header=None).values.flatten()
    
    # Leer datos reducidos
    X = pd.read_csv(os.path.join(script_dir, 'Xtrain_reduced.csv'), header=None).values.T
    
    # Cargar etiquetas
    Y = pd.read_csv(os.path.join(data_path, 'classtrain.csv'), header=None).values.T
    
    param = {
        'MaxIter': int(config[0]),
        'mu': float(config[1]),
        'BatchSize': int(config[2]),
        'L2_lambda': 0.0001  # Regularización L2
    }
    
    print(f"Input shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")
    print(f"Training parameters: {param}")
    print(f"\n{'='*50}")
    print("Training Softmax classifier on REDUCED data...")
    print(f"{'='*50}\n")
    
    W, Cost = train_softmax(X, Y, param)
    
    # Guardar resultados
    np.save(os.path.join(script_dir, 'W_softmax.npy'), W)
    
    cost_history_df = pd.DataFrame({'Cost': Cost})
    cost_history_df.to_csv(os.path.join(script_dir, 'cost_history.csv'), index=False)
    
    print(f"\n{'='*50}")
    print(f"Training complete. Final Cost: {Cost[-1]:.6f}")
    print("Weights saved to 'W_softmax.npy'")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()