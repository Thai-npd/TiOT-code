import numpy as np
import ot
from scipy.optimize import linprog




def normalization(x,y):
    """
    Normalize input time series by formular
    x = (x - mean(x)) / std(x)

    Parameters
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)

    Returns:
        (tuple): normalized x, normalized y
    """

    return (x - np.mean(x)) / np.std(x), (y - np.mean(y)) / np.std(y)

def costmatrix0(x ,y, w) -> np.ndarray : 
    """
    Cost matrix define in the TAOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """

    n, m = len(x), len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = np.average((x[i] - y[j])**2) + w * (t[i] - s[j])**2
    M = M / np.median(M)
    return M   

def costmatrix1(x,y, w) -> np.ndarray :
    """
    Cost matrix define in the TiOT's way: C.

    Parameters:
        x (ndarray): 1D array of shape (n,)
        y (ndarray): 1D array of shape (m,)
        w (float)  : parameter to calculate C_ij

    Returns:
        ndarray: 2D array of shape (n,m)
    """

    n, m = len(x), len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = w * np.linalg.norm(x[i] - y[j])**2 + (1-w) * (t[i] - s[j])**2
    
    return M  

def TiOT(x, y, a = None, b = None, detail_mode = False):
    """
    Solve the Time-integrated Optimal Transport (TiOT) problem between two discrete distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        detail_mode (bool, optional): If False (default), returns distance, optimal w
                                      If True, returns distance, transport plan, optimal w

    Returns:
        tuple: 
            - If `detail_mode=False`: returns a transport plan matrix of shape (n, m).
            - If `detail_mode=True`: returns a dictionary containing the transport plan and additional details.

    Notes:
        - The input distributions `a` and `b` must sum to 1 if provided.
    """

    n, m = len(x), len(y)
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
    res = linprog(c, A_ub=A, b_ub=r, bounds=bounds)
    if detail_mode == False:
        return -res.fun, res.x[-1]
    else:
        return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1]



def eTiOT(x, y, a = None, b = None, eps = 0.01, maxIter = 5000, tolerance = 0.005, solver = 'newton', subprob_tol = 10**-7, w_update_freq = 1, verbose = False):
    """
    Solves the entropic Time-integrated Optimal Transport (eTiOT) problem use block coordinate descent.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        eps (float, optional): Entropic regularization parameter. Default is 0.01.
        maxIter (int, optional): Maximum number of iterations for BCD. Default is 5000.
        tolerance (float, optional): Convergence tolerance for BCD. Default is 0.005.
        solver (str, optional): Method used for finding optimal w. Default is 'newton'.
        subprob_tol (float, optional): Tolerance for solving the inner subproblem (e.g., newton). Default is 1e-7.
        w_update_freq (int, optional): Frequency (in iterations) at which the weight variable `w` is updated. Default is 1.
        verbose (bool, optional): If True, prints progress. Default is False.

    Returns:
        tuple: distance, transport plan, optimal w
    """

    n, m = len(x), len(y)
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

    def newton(g,h, w, subprob_tol = 10**-7, maxIter = 10):
        def f(w):
            C = w*value_diff + (1-w)*time_diff
            K = np.exp(-C/eps)
            return eps * (g.T @ K @ h)
        
        def df12(w):
            nC =  w * TV - time_diff 
            K = np.exp(nC/(eps))
            df = g.T @ ((TV * K) @ h)
            df2 = (1/eps) * g.T @ (((TV**2) * K) @ h)
            return df, df2 
        
        for i in range(maxIter):
            dfw, df2w = df12(w)
            w = w - dfw / df2w
            if np.abs(dfw) < subprob_tol or w >=1  or w <= 0 :
                if verbose == True: 
                    print(f"Newton Algorithm converges after {i+1} iterations")
                if 0 < w < 1: 
                    break
                elif w >= 1: 
                    w = 1
                    break
                else:
                    w = 0
                    break
                
        if verbose == True:
            if i == maxIter: print(f"Newton algorithm does not converge after {i} iterations")
        C = w*value_diff + (1-w)*time_diff
        K = np.exp(-C/eps)
        return w, K
    
    g, h, w = np.ones(n)/n , np.ones(m)/m , 0.5
    C = w*value_diff + (1-w)*time_diff
    K = np.exp(-C/eps)
    curIter = 0
    if solver == 'newton': solver = newton
    else:
        print("For now, no other solvers is available! Choose 'newton' by default. ")
        solver = newton

    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if curIter % w_update_freq ==0 :
            w,  K = solver(g,h, w, subprob_tol= subprob_tol)
        if curIter % 20 == 0:
            if np.sum(np.abs(g * (K @ h) - a)) < tolerance : 
                if verbose == True: 
                    print(f"TiOT-BCD Algorithm converges after {curIter+1} iterations")
                break
        curIter += 1
    if verbose == True:
        if curIter == maxIter: print(f"TiOT algorithm did not stop after {maxIter} iterations")
    C = w*value_diff + (1-w)*time_diff
    transport_plan = np.diag(g) @ K @ np.diag(h)
    return np.sum(C * transport_plan), transport_plan, w, 


def TAOT(x, y, a = None, b = None, w = 0.5, costmatrix = costmatrix1):
    """
    Solve the Time Adaptive Optimal Transport (TAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        costmatrix (callable, optional): A function to compute the cost matrix between x and y. Default is `costmatrix1`.

    Returns:
        tuple: distance, transport_plan
    """
    n, m = len(x), len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    result = ot.lp.emd2(a,b,M, return_matrix=True) # ot.emd2(a,b,M, return_matrix=True)
    distance = result[0]
    transport_plan = result[1]['G']
    return distance, transport_plan

def eTAOT(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1,  maxIter=5000, tolerance=0.005, verbose = False):
    """
    Solves the entropic Time Adaptive Optimal Transport (eTAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        eps (float, optional): Entropic regularization parameter. Default is 0.01.
        costmatrix (callable, optional): Function to compute the cost matrix between `x` and `y`. Default is `costmatrix1`.
        maxIter (int, optional): Maximum number of iterations for BCD. Default is 5000.
        tolerance (float, optional): Convergence tolerance for BCD. Default is 0.005.
        verbose (bool, optional): If True, prints progress and debug information. Default is False.
                
    Returns:
        tuple: distance, transport_plan
    """
    n, m = len(x), len(y)
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
                if verbose == True: 
                    print(f"TAOT-BCD Algorithm converges after {curIter+1} iterations")
                break
        curIter += 1
    if verbose == True:
        if curIter == maxIter: print(f"TAOT algorithm did not stop after {maxIter} iterations")
    transport_plan = np.diag(g) @ K @ np.diag(h)
    distance = np.sum(M * transport_plan)
    
    return distance, transport_plan


