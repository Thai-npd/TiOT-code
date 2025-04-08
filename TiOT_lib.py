import numpy as np
import ot
import pandas as pd
from collections import defaultdict
import seaborn as sns
from scipy.optimize import linprog
import sys
import matplotlib.pyplot as plt
import time
import scipy
import pulp

def multi_dimension_reshape(series):
    arr = []
    if isinstance(series[0], list):
        for j in range(len(series[0])):
            arr.append([])
            for i in range(len(series)):
                arr[j].append(series[i][j])
        return np.array(arr)
    else:
        return series


def normalization(x,y):
    return (x - np.mean(x)) / np.std(x), (y - np.mean(y)) / np.std(y)

def costmatrix0(x,y, w):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = np.average((x[i] - y[j])**2) + w * (t[i] - s[j])**2
    M = M / np.median(M)
    return M   

def costmatrix1(x,y, w):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = w * np.linalg.norm(x[i] - y[j])**2 + (1-w) * (t[i] - s[j])**2
    
    return M  

def TiOT(x, y, a = None, b = None, detail_mode = False):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    if a == None:
        a = [1/n for i in range(n)]
    if b == None:
        b = [1/m for j in range(m)]
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)    
    I = -np.eye(n)
    J = -np.eye(m)
    A = np.array([np.concatenate((I[i], J[j])) for i in range(n) for j in range(m)])
    extraCol = np.array([[(t[i] - s[j])**2 - np.linalg.norm(x[i] - y[j])**2 for i in range(n) for j in range(m)]])
    A = np.hstack((A, extraCol.T))
    r = np.array([(t[i] - s[j])**2 for i in range(n) for j in range(m)])
    c = np.array(np.concatenate((a,b, [0])))
    bounds = [(None, None) for var in range(m+n)] + [(0,1)]
    #print(f"Size checking A: {A.shape}, b = {r.shape}, c = {c.shape}")
    res = linprog(c, A_ub=A, b_ub=r, bounds=bounds)
    x_opt = res.x
    u = x_opt[:int(len(x_opt) / 2)]
    v = x_opt[int(len(x_opt) / 2):-1]
    w = x_opt[-1]
    if detail_mode == False:
        return -res.fun, res.x[-1]
    else:
        return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1]

def eTiOT(x, y, a = None, b = None, eps = 0.01, maxIter = 5000, tolerance = 0.005, t0 = 0.01, solver = 'newton', tol = 10**-7, w_update_freq = 1):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)
    value_diff = np.zeros((n,m))
    time_diff = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            value_diff[i,j] = np.linalg.norm(x[i] - y[j])**2
            time_diff[i,j] = (t[i] - s[j])**2
    TV = time_diff - value_diff
    def PGD(g,h, w, draw = False,t = 0.01, tol = 10**-7, maxIter = 100):
        def f(w):
            C = w*value_diff + (1-w)*time_diff
            K = np.exp(-C/eps)
            return eps * (g.T @ K @ h)
        
        def df(w):
            nC =  w * TV - time_diff 
            K = np.exp(nC/(eps))
            return g.T @ ((TV * K) @ h)
        
        def prox_g(w, stepsize):
            if w > 1:
                w = 1
            elif w < 0:
                w = 0
            return w 

        for i in range(maxIter):
            w_prev = w
            w = w - t*df(w)
            w = prox_g(w, t)
            if np.abs(w-w_prev) < tol:
                print(f"PGD Algorithm converges after {i+1} iterations with w = {w} and df = {df(w)} and {np.abs(w-w_prev)}")
                break
        if i == maxIter: print(f"PGD Algorithm does not converge after {i} iterations")
        C = w*value_diff + (1-w)*time_diff
        K = np.exp(-C/eps)
        return w, K

    def newton(g,h, w, draw = False,t = 0.01, tol = 10**-7, maxIter = 100):
        def f(w):
            C = w*value_diff + (1-w)*time_diff
            K = np.exp(-C/eps)
            return eps * (g.T @ K @ h)
        
        def df(w):
            nC =  w * TV - time_diff 
            K = np.exp(nC/(eps))
            return g.T @ ((TV * K) @ h)
        
        def df2(w):
            nC =  w * TV - time_diff 
            K = np.exp(nC/(eps))
            return  (1/eps) * g.T @ (((TV**2) * K) @ h)
        
        def df12(w):
            nC =  w * TV - time_diff 
            K = np.exp(nC/(eps))
            return g.T @ ((TV * K) @ h), (1/eps) * g.T @ (((TV**2) * K) @ h)
        
        for i in range(maxIter):
            w_prev = w
            dfw, df2w = df12(w)
            w = w - dfw / df2w
            if np.abs(dfw) < tol or w >=1  or w <= 0 :
                print(f"Newton Algorithm converges after {i+1} iterations with w = {w} and df = {dfw}")
                if 0 < w < 1:
                    break
                elif w >= 1: 
                    w = 1
                    break
                else:
                    w = 0
                    break
                
        if i == maxIter: print(f"Newton algorithm does not converge after {i} iterations")
        C = w*value_diff + (1-w)*time_diff
        K = np.exp(-C/eps)
        return w, K
    
    g = np.ones(n) / n
    h = np.ones(m) / m
    w = 0.5
    C = w*value_diff + (1-w)*time_diff
    K = np.exp(-C/eps)

    curIter = 0
    if solver == 'newton': solver = newton
    else:
        solver = PGD
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if curIter % w_update_freq ==0 :
            w,  K = solver(g,h, w, tol= tol, t=t0)
        if curIter % 20 == 0:
            if np.sum(np.abs(g * (K @ h) - a)) < tolerance : #and np.linalg.norm(w - w_prev) < tolerance
                #print(f"BCD Algorithm converges after {curIter} iterations")
                break
        curIter += 1
    #if curIter == maxIter: print(f"Algorithm did not stop after {maxIter} iterations")
    C = w*value_diff + (1-w)*time_diff
    transport_plan = np.diag(g) @ K @ np.diag(h)
    print(f"optimal w = {w}, optimal value : {np.sum(C * transport_plan)}")
    return np.sum(C * transport_plan), transport_plan, w, 


def TAOT(x, y, a = None, b = None, w = 0.5, costmatrix = costmatrix1, apply_MA = False):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    result = ot.emd2(a,b,M, return_matrix=True)
    distance = result[0]
    transport_plan = result[1]['G']
    return distance, transport_plan

def eTAOT(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1,  maxIter=5000, tolerance=0.005):
    x = multi_dimension_reshape(x)
    y = multi_dimension_reshape(y)
    n = len(x)
    m = len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    K = np.exp(-M/eps)
    g = np.ones(n) / n
    h = np.ones(m) / m
    curIter = 0
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        
        if curIter % 20 == 0:
            criterion = np.sum(np.abs(g * (K @ h) - a))
            if criterion < tolerance:
                break
        curIter += 1
    transport_plan = np.diag(g) @ K @ np.diag(h)
    distance = np.sum(M * transport_plan)
    
    return distance, transport_plan


