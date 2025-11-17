# Dimensionality Reducing 
import pandas as pd
import numpy as np
from utility import *


def pc_svd(X, K3):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V3 = Vt[:K3, :].T
    return V3


def updW_adam(W1, V, S, gW1, mu, n):
    V = mu * V + (1 - mu) * gW1
    S = mu * S + (1 - mu) * (gW1 ** 2)
    V_hat = V / (1 - mu ** n)
    S_hat = S / (1 - mu ** n)
    W1 = W1 - (eta * V_hat) / (np.sqrt(S_hat) + epsilon)
    return W1, V, S


def gradW1(Act, W2):
    Err = Act - np.dot(W2, Act)
    gW1 = -2 * np.dot(Err, Act.T)
    return gW1


def forward(xe, W1, W2):
    Act = sigmoid(np.dot(W1, xe))
    Xr = np.dot(W2, Act)
    Cost = np.sum((xe - Xr) ** 2)
    return Act, Cost


def w2_pinv(xe, W1, C):
    Act = sigmoid(np.dot(W1, xe))
    W2 = np.dot(xe, np.linalg.pinv(Act))
    return W2


def train_minibatch(Xe, W1, W2, V, S, param):
    numBatch = calcula_number_batch()
    Cost = []
    for n in range(1, numBatch + 1):
        xe = get_miniBatch(n, Xe, param.BatchSize)
        W2 = w2_pinv(xe, W1, param.C)
        Act, cost = forward(xe, W1, W2)
        gW1 = gradW1(Act, W2)
        W1, V, S = updW_adam(W1, V, S, gW1, param.mu, n)
        Cost.append(cost)
    
    MSEavg = np.mean(Cost)
    return MSEavg, W1, V, S, W2


def train_sae(X, param):
    X = zscores_dataset(X)
    W1, W2 = randW()
    V, S = zeros()
    
    for Iter in range(1, param.MaxIter + 1):
        Xe = rand_position_dataset(X)
        MSE_iter, W1, W2, V, S = train_minibatch(Xe, W1, W2, V, S, param)
        if Iter % 50 == 0:
            print(f"MSE({Iter})")
    
    return W1


def randW(Nprev, Nnext):
    r = np.sqrt(6 / (Nprev + Nnext))
    W = np.random.rand(Nprev, Nnext) * 2 * r - r
    return W


def main():
    X = load_dataset()
    param = load_param()
    X = zscores_dataset(X)
    
    for i in range(1, param.NumAE + 1):
        Vr_i = train_sae(X, param)
        X = act_sigmoid(np.dot(Vr_i, X))
    
    V3 = pc_svd(X, param.K3)
    X = act_sigmoid(np.dot(V3.T, X))
    Vr_i = V3
    
    save_new_data(X, Vr_i)


if __name__ == '__main__':
    main()
