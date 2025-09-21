import numpy as np
import ot
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye, kron
#np.seterr(divide='ignore', invalid='ignore', over='ignore')
import time
import warnings
#import cProfile
from scipy.stats import norm

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

def costmatrix0_old(x ,y, w) -> np.ndarray : 
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

    spatial_cost, temporal_cost = spat_temp_cost(x,y)
    C = spatial_cost + w * temporal_cost
    C = C / np.median(C)
    return C  

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
    x,y = normalization(x,y)
    spatial_cost, temporal_cost = spat_temp_cost(x,y)
    return w * spatial_cost + (1-w)*temporal_cost

def costmatrix1old(x,y, w) -> np.ndarray :
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

def spat_temp_cost(x,y, eps = 1):
    n, m = len(x), len(y)
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    if x.ndim == 1 and y.ndim == 1:
        spatial_cost =  (x[:, None] - y[None, :])**2 / eps
    else:
        x_norm2 = np.sum(x**2, axis=1)[:, None]
        y_norm2 = np.sum(y**2, axis=1)[None, :]
        spatial_cost = (x_norm2 + y_norm2 - 2 * x @ y.T) / eps
    temporal_cost =  (t[:, None] - s[None, :])**2 / eps
    return spatial_cost, temporal_cost
 
def TiOTold(x, y, a = None, b = None, detail_mode = False, verbose = False, timing = False):
    """
    Solve the Time-integrated Optimal Transport (TiOT) problem between two discrete distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        detail_mode (bool, optional): If False (default), returns distance, optimal w
                                      If True, returns distance, transport plan, optimal w
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: 
            - If `detail_mode=False`: returns distance and optimal w.
            - If `detail_mode=True`: returns distance, transport plan and optimal w.

    Notes:
        - The input distributions `a` and `b` must sum to 1 if provided.
    """

    start = time.perf_counter() 
    n, m = len(x), len(y)
    if a == None:
        a = [1/n for i in range(n)]
    if b == None:
        b = [1/m for j in range(m)]
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)    
    I = -np.eye(n)
    J = -np.eye(m)
    A = np.array([np.concatenate((I[i], J[j])) for i in range(n) for j in range(m)], dtype=np.float32)
    extraCol = np.array([[(t[i] - s[j])**2 - np.linalg.norm(x[i] - y[j])**2 for i in range(n) for j in range(m)]])
    A = np.hstack((A, extraCol.T))
    r = np.array([(t[i] - s[j])**2 for i in range(n) for j in range(m)])
    c = np.array(np.concatenate((a,b, [0])))
    bounds = [(None, None) for var in range(m+n)] + [(0,1)]
    res = linprog(c, A_ub=A, b_ub=r, bounds=bounds)
    end = time.perf_counter()

    if timing == True:
        if detail_mode == False:
            return -res.fun, res.x[-1], end - start
        else:
            return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1], end -start
    else:
        if detail_mode == False:
            return -res.fun, res.x[-1]
        else:
            return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1]

# def TiOTnew(x, y, a = None, b = None, detail_mode = False, verbose = False, timing = False):
#     """
#     Solve the Time-integrated Optimal Transport (TiOT) problem between two discrete distributions.

#     Parameters:
#         x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
#         y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
#         a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
#         b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
#         detail_mode (bool, optional): If False (default), returns distance, optimal w
#                                       If True, returns distance, transport plan, optimal w
#         timing (bool, optinal): If True, record the elapsed time and return it

#     Returns:
#         tuple: 
#             - If `detail_mode=False`: returns distance and optimal w.
#             - If `detail_mode=True`: returns distance, transport plan and optimal w.

#     Notes:
#         - The input distributions `a` and `b` must sum to 1 if provided.
#     """

#     start = time.perf_counter() 
#     n, m = len(x), len(y)
#     if a == None: a = np.ones(n) / n
#     if b == None: b = np.ones(m) / m
#     t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
#     x,y = normalization(x,y)   
#     A = np.hstack([
#     -np.kron(np.eye(n, dtype=np.float32), np.ones((m,1), dtype=np.float32)),
#     -np.kron(np.ones((n,1), dtype=np.float32), np.eye(m, dtype=np.float32))
# ])
#     spatial_cost, temporal_cost = spat_temp_cost(x,y)
#     extraCol = (temporal_cost - spatial_cost).reshape(-1, 1)
#     A = np.hstack((A, extraCol))
#     dense_mem = A.nbytes
#     print("Dense memory:", dense_mem/1e6, "MB")
#     c = np.array(np.concatenate((a,b, [0])))
#     bounds = [(None, None) for var in range(m+n)] + [(0,1)]
#     res = linprog(c, A_ub=A, b_ub=temporal_cost.flatten(), bounds=bounds)
#     end = time.perf_counter()

#     if timing == True:
#         if detail_mode == False:
#             return -res.fun, res.x[-1], end - start
#         else:
#             return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1], end -start
#     else:
#         if detail_mode == False:
#             return -res.fun, res.x[-1]
#         else:
#             return -res.fun, TAOT(x,y, w = res.x[-1])[1], res.x[-1]
 
def TiOT(x, y, a=None, b=None, detail_mode=False, verbose=False, timing=False):
    """
    Solve the Time-integrated Optimal Transport (TiOT) problem between two discrete distributions.
    Uses sparse matrices for better scalability.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        detail_mode (bool, optional): If False (default), returns distance, optimal w.
                                      If True, returns distance, transport plan, optimal w.
        timing (bool, optional): If True, record elapsed time and return it.

    Returns:
        tuple: 
            - If `detail_mode=False`: returns distance and optimal w.
            - If `detail_mode=True`: returns distance, transport plan, optimal w.
            - If `timing=True`: additionally return elapsed time.
    """

    start = time.perf_counter()
    n, m = len(x), len(y)

    # Default uniform weights if not provided
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    # Normalize time indices (dummy normalization function assumed available)
    t, s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x, y = normalization(x, y)

    # Sparse constraint blocks
    A1 = -kron(eye(n, format="csr"), np.ones((m,1), dtype=np.float32))   # - kron(I_n, 1_m)
    A2 = -kron(np.ones((n,1), dtype=np.float32), eye(m, format="csr"))   # - kron(1_n, I_m)

    # Spatial and temporal costs (assumed provided)
    spatial_cost, temporal_cost = spat_temp_cost(x, y)

    # Extra column (sparse version)
    extraCol = (temporal_cost - spatial_cost).reshape(-1, 1)
    extraCol = csr_matrix(extraCol)

    # Combine everything into one sparse constraint matrix
    A = hstack([A1, A2, extraCol], format="csr")
    sparse_mem = A.data.nbytes + A.indptr.nbytes + A.indices.nbytes
    # Cost vector
    c = np.concatenate((a, b, [0]))

    # Bounds: (None, None) for the transport vars, (0,1) for w
    bounds = [(None, None)] * (n + m) + [(0, 1)]

    # Solve with HiGHS
    res = linprog(c, A_ub=A, b_ub=temporal_cost.flatten(), bounds=bounds, method="highs")

    end = time.perf_counter()

    # Handle outputs
    if timing:
        if not detail_mode:
            return -res.fun, res.x[-1], end - start
        else:
            return -res.fun, TAOT(x, y, w=res.x[-1])[1], res.x[-1], end - start
    else:
        if not detail_mode:
            return -res.fun, res.x[-1]
        else:
            return -res.fun, TAOT(x, y, w=res.x[-1])[1], res.x[-1]       

def eTiOTold(x, y, a = None, b = None, eps = 0.01, maxIter = 5000, tolerance = 0.005, solver = 'PGD', eta = 10**-2, init_stepsize = True, submax_iter = 50,  subprob_tol = 10**-7, freq = 20, verbose = False, timing = False):
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
        freq (int, optional): Frequency (in iterations) at which the weight variable `w` is updated. Default is 1.
        verbose (bool, optional): If True, prints progress. Default is False.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport plan, optimal w
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    t,s = normalization(np.arange(1, n+1), np.arange(1, m+1))
    x,y = normalization(x,y)
    spatial_cost = np.zeros((n,m))
    temporal_cost = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            spatial_cost[i,j] = np.linalg.norm(x[i] - y[j])**2
            temporal_cost[i,j] = (t[i] - s[j])**2
    TV = (temporal_cost - spatial_cost)

    def newton(g,h, w, subprob_tol = 10**-7, maxIter = 10, eta = None, init_stepsize = None):
        def f(w):
            C = w*spatial_cost + (1-w)*temporal_cost
            K = np.exp(-C/eps)
            return eps * (g.T @ K @ h)
        
        def df12(w):
            nC =  w * TV - temporal_cost 
            K = np.exp(nC/(eps))
            df = g.T @ ((TV * K) @ h)
            df2 = (1/eps) * g.T @ (((TV**2) * K) @ h)
            return df, df2 
        for i in range(maxIter):
            dfw, df2w = df12(w)
            #if i>=1:print(f'possible stepsize  {np.linalg.norm(w- w_old)/ np.linalg.norm(dfw - dfw_old) } compare to 1/dfw2 = {1/df2w}' )
            w = w - dfw / df2w
            if np.abs(dfw) < subprob_tol or w >=1  or w <= 0 :
                if verbose > 1: 
                    print(f"Newton Algorithm converges after {i+1} iterations with w = {w}, df = {dfw}, df2 = {df2w} and possible stepsize 1/df2 = {1/df2w}")
                if 0 < w < 1: 
                    break
                elif w >= 1: 
                    w = 1
                    break
                else:
                    w = 0
                    break
                
        if verbose > 1:
            if i == maxIter: print(f"Newton algorithm does not converge after {i} iterations")
        C = w*spatial_cost + (1-w)*temporal_cost
        K = np.exp(-C/eps)
        return w, K
    


    def PGD(g,h, w, subprob_tol = 10**-7, maxIter = 50, eta = 10**-2, init_stepsize = False):
        def f(w):
            C = w*spatial_cost + (1-w)*temporal_cost
            K = np.exp(-C/eps)
            return  -eps * np.log(g) @ a - eps * np.log(h) @ b + eps * (g.T @ K @ h)
        def df(w):
            nC =  w * TV - temporal_cost
            K = np.exp(nC/(eps))
            return g.T @ ((TV * K) @ h)
        def df2w(w):
            nC =  w * TV - temporal_cost 
            K = np.exp(nC/(eps))
            df2 = (1/eps) * g.T @ (((TV**2) * K) @ h)
            return df2
        def proj(w):
            if w > 1:
                w = 1
            elif w < 0:
                w = 0
            return w 
        def init_eta(df2w):
            possible_stepsize = 1/df2w
            if possible_stepsize >= 10:
                return possible_stepsize/20
            else:
                return possible_stepsize/10
        if init_stepsize:
            eta = init_eta(df2w(w))

        for i in range(maxIter):
            w_prev = w
            dfw = df(w)
            w = proj(w - eta*dfw)
            if np.abs(w-w_prev) < subprob_tol:
                break
        #print(f"Total subiteration needed for PGD: {i} with df = {df(w)}")
        C = w*spatial_cost + (1-w)*temporal_cost
        K = np.exp(-C/eps)
        if i == maxIter - 1 and verbose >= 1 : print(f"PGD Algorithm does not converge after {i} iterations with stepsize {eta} and dfw = {dfw}") # and verbose >= 1
        return w, K

    g, h, w = np.ones(n)/n , np.ones(m)/m , 0.5
    C = w*spatial_cost + (1-w)*temporal_cost
    K = np.exp(-C/eps)
    curIter = 0
    if solver == 'newton': 
        solver = newton
    else:
        solver = PGD
    while curIter < maxIter:
        # g_old = g
        # h_old = h
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if curIter % freq ==0 :
            w,  K = solver(g,h, w, subprob_tol= subprob_tol, maxIter=submax_iter, eta=eta, init_stepsize=init_stepsize)
        if np.any(np.isnan(g)) or np.any(np.isnan(h)) or np.linalg.norm(g) == 0:
            warnings.warn(f"Warning: numerical errors at iteration {curIter}: consider larger epsilon or smaller stepsize(current eta = {eta})")
            # g = g_old
            # h = h_old
            break
        if curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance: # np.linalg.norm(g - g_old) / np.linalg.norm(g_old)
            # print(f"Stop where {np.sum(np.abs(g * (K @ h) - a))}")
            if verbose >= 1:
                print(f"TiOT-BCD Algorithm converges after {curIter+1} iterations ")
            break
        curIter += 1
    if curIter == maxIter and verbose >= 1: print(f"TiOT algorithm did not stop after {maxIter} iterations")
    C = w*spatial_cost + (1-w)*temporal_cost
    transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1)) # np.diag(g) @ K @ np.diag(h)
    distance = np.sum(C * transport_plan)
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, w, end - start
    else:
        return distance, transport_plan, w

def eTiOT(x, y, a = None, b = None, eps = 0.01, maxIter = 5000, tolerance = 0.005, solver = 'PGD', eta = 10**-2, init_stepsize = True, submax_iter = 50,  subprob_tol = 10**-7, freq = 20, verbose = False, timing = False):
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
        freq (int, optional): Frequency (in iterations) at which the weight variable `w` is updated. Default is 1.
        verbose (bool, optional): If True, prints progress. Default is False.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport plan, optimal w
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    x,y = normalization(x,y)
    spatial_cost, temporal_cost = spat_temp_cost(x,y,eps)
    spat_temp_diff = temporal_cost - spatial_cost

    def newton(g,h, w, subprob_tol = 10**-7, maxIter = 10, eta = None, init_stepsize = None):
        def f(w):
            C = w*spatial_cost + (1-w)*temporal_cost
            K = np.exp(-C/eps)
            return eps * (g.T @ K @ h)
        
        def df12(w):
            nC =  w * spat_temp_diff - temporal_cost 
            K = np.exp(nC/(eps))
            df = g.T @ ((spat_temp_diff * K) @ h)
            df2 = (1/eps) * g.T @ (((spat_temp_diff**2) * K) @ h)
            return df, df2 
        for i in range(maxIter):
            dfw, df2w = df12(w)
            #if i>=1:print(f'possible stepsize  {np.linalg.norm(w- w_old)/ np.linalg.norm(dfw - dfw_old) } compare to 1/dfw2 = {1/df2w}' )
            w = w - dfw / df2w
            if np.abs(dfw) < subprob_tol or w >=1  or w <= 0 :
                if verbose > 1: 
                    print(f"Newton Algorithm converges after {i+1} iterations with w = {w}, df = {dfw}, df2 = {df2w} and possible stepsize 1/df2 = {1/df2w}")
                if 0 < w < 1: 
                    break
                elif w >= 1: 
                    w = 1
                    break
                else:
                    w = 0
                    break
                
        if verbose > 1:
            if i == maxIter: print(f"Newton algorithm does not converge after {i} iterations")
        C = w*spatial_cost + (1-w)*temporal_cost
        K = np.exp(-C/eps)
        return w, K
    
    def PGD(g,h, w, subprob_tol = 10**-7, maxIter = 50, eta = 10**-2, init_stepsize = False):
        def f(w):
            C = w*spatial_cost + (1-w)*temporal_cost
            K = np.exp(-C/eps)
            return  -eps * np.log(g) @ a - eps * np.log(h) @ b + eps * (g.T @ K @ h)
        def df(w):
            # The formulation for the derivative is df(w) = g.T @ (spat_temp_diff * K_w) @ h. To optimize the efficiency the variable temp(temporary) is used through out the entire process.
            temp = w * spat_temp_diff
            temp -= temporal_cost
            np.exp(temp, out= temp)
            temp *= spat_temp_diff
            return eps *( g.T @ temp @ h)
        
        def df2w(w):
            temp = w * spat_temp_diff
            temp -= temporal_cost
            np.exp(temp, out= temp)
            temp *= spat_temp_diff**2
            return  eps* (g.T @ (temp @ h))
        
        def proj(w):
            if w > 1:
                w = 1
            elif w < 0:
                w = 0
            return w 
        def init_eta(df2w):
            possible_stepsize = 1/df2w
            if possible_stepsize >= 10:
                return possible_stepsize/20
            else:
                return possible_stepsize/10
        if init_stepsize:
            eta = init_eta(df2w(w))

        for i in range(maxIter):
            w_prev = w
            dfw = df(w)
            w = proj(w - eta*dfw)
            if np.abs(w-w_prev) < subprob_tol:
                break
        #print(f"Total subiteration needed for PGD: {i} with df = {df(w)}")
        K = w * spat_temp_diff
        K -= temporal_cost
        np.exp(K, out = K) 
        # The three lines above is to compute K efficiently with the first two lines computing the cost negative C and the third line compute exp(nC).
        if i == maxIter - 1 and verbose >= 1 : print(f"PGD Algorithm does not converge after {i} iterations with stepsize {eta} and dfw = {dfw}") # and verbose >= 1
        return w, K

    g, h, w = np.ones(n)/n , np.ones(m)/m , 0.5
    nC = w * spat_temp_diff - temporal_cost
    K = np.exp(nC)
    curIter = 0
    if solver == 'newton': 
        solver = newton
    else:
        solver = PGD
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if curIter % freq ==0 :
            w, K = solver(g,h, w, subprob_tol= subprob_tol, maxIter=submax_iter, eta=eta, init_stepsize=init_stepsize)
        if np.any(np.isnan(g)) or np.any(np.isnan(h)):
            warnings.warn(f"Warning: numerical errors at iteration {curIter}: consider larger epsilon or smaller stepsize(current eta = {eta})")
            break
        if curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance: 
            if verbose >= 1:
                print(f"TiOT-BCD Algorithm converges after {curIter+1} iterations ")
            break
        curIter += 1
    if curIter == maxIter and verbose >= 1: print(f"TiOT algorithm did not stop after {maxIter} iterations")
    C = eps * (temporal_cost - w*spat_temp_diff)
    K *= h.reshape((1, -1)) 
    K *= g.reshape((-1,1))
    #transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1)) 
    distance = np.einsum("ij,ij->", C, K)
    end = time.perf_counter()
    if timing == True:
        return distance, K, w, end - start
    else:
        return distance, K, w

def TAOTold(x, y, a = None, b = None, w = 0.5, costmatrix = costmatrix1old, verbose = None, timing = False):
    """
    Solve the Time Adaptive Optimal Transport (TAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        costmatrix (callable, optional): A function to compute the cost matrix between x and y. Default is `costmatrix1`.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    result = ot.lp.emd2(a,b,M, numItermax = 10**8, return_matrix=True) 
    distance = result[0]
    transport_plan = result[1]['G']
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end -start
    else:
        return distance, transport_plan

def TAOT(x, y, a = None, b = None, w = 0.5, costmatrix = costmatrix1, verbose = None, timing = False):
    """
    Solve the Time Adaptive Optimal Transport (TAOT) problem between two empirical distributions.

    Parameters:
        x (ndarray): Array of shape (n, d) or (n,) representing n source points in d-dimensional space.
        y (ndarray): Array of shape (m, d) or (m,) representing m target points in d-dimensional space.
        a (ndarray, optional): 1D array of shape (n,) representing source weights (default: uniform weights). 
        b (ndarray, optional): 1D array of shape (m,) representing target weights (default: uniform weights).
        w (float, optional): Weight parameter to compute cost matrix C.
        costmatrix (callable, optional): A function to compute the cost matrix between x and y. Default is `costmatrix1`.
        timing (bool, optinal): If True, record the elapsed time and return it

    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    M = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    result = ot.lp.emd2(a,b,M, numItermax = 10**8, return_matrix=True) 
    distance = result[0]
    transport_plan = result[1]['G']
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end -start
    else:
        return distance, transport_plan

def eTAOTold(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1old,  maxIter=5000, tolerance=0.005, freq = 20, verbose = False, timing = False):
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
        timing (bool, optinal): If True, record the elapsed time and return it
    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
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
        if  curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance:
            if verbose >= 1: 
                print(f"TAOT-BCD Algorithm converges after {curIter+1} iterations")
            break
        curIter += 1
    if verbose >= 1:
        if curIter == maxIter: print(f"TAOT algorithm did not stop after {maxIter} iterations")
    transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1))
    distance = np.sum(M * transport_plan)
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end - start
    else:
        return distance, transport_plan

def eTAOT(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1,  maxIter=5000, tolerance=0.005, freq = 20, verbose = False, timing = False):
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
        timing (bool, optinal): If True, record the elapsed time and return it
    Returns:
        tuple: distance, transport_plan
    """
    start = time.perf_counter()
    n, m = len(x), len(y)
    C = costmatrix(x, y, w)
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    K = np.exp(-C/eps)
    g = np.ones(n) / n
    h = np.ones(m) / m
    curIter = 0
    while curIter < maxIter:
        g = a/ (K @ h)
        h = b/(K.T @ g)
        if  curIter % freq == 0 and np.sum(np.abs(g * (K @ h) - a)) < tolerance:
            if verbose >= 1: 
                print(f"TAOT-BCD Algorithm converges after {curIter+1} iterations")
            break
        curIter += 1
    if verbose >= 1:
        if curIter == maxIter: print(f"TAOT algorithm did not stop after {maxIter} iterations")
    transport_plan = g.reshape((-1, 1)) * K * h.reshape((1, -1))
    distance = np.sum(C * transport_plan)
    distance = np.einsum("ij,ij->", C, transport_plan)
    end = time.perf_counter()
    if timing == True:
        return distance, transport_plan, end - start
    else:
        return distance, transport_plan


def sinkhorn(x, y, a = None, b = None, w = 0.5, eps = 0.01, costmatrix = costmatrix1,  maxIter=5000, tolerance=0.005, freq = 20, verbose = False):
    n, m = len(x), len(y)
    M = costmatrix(x, y, w)
    start = time.perf_counter()
    if a == None: a = np.ones(n) / n
    if b == None: b = np.ones(m) / m
    transport_plan, log = ot.bregman.sinkhorn_knopp(a,b,M, reg  = eps, verbose = True, log = True)
    distance = np.sum(M * transport_plan)
    print(type(M), type(a), type(b), type(log['u']))
    end = time.perf_counter()

    return distance, transport_plan, end - start

def df_test(w, TV, temporal_cost, g, h, eps):
    nC = w * TV - temporal_cost
    #C = temporal_cost - w*TV
    K = np.exp(nC / eps)
    return g.T @ ((TV*K) @ h)

def df_test_loop(w, spat_temp_diff, temporal_cost, g, h, eps):
    temp = w * spat_temp_diff
    temp -= temporal_cost
    np.exp(temp, out= temp)
    temp *= spat_temp_diff
    return eps * (g.T @ temp @ h)


def time_analyse1():
    np.random.seed(41)

    # Matrix and vector sizes
    n, m = 2000, 2000

    # Random matrices
    TV = np.random.rand(n, m)
    temporal_cost = np.random.rand(n, m)

    # Random vectors
    g = np.random.rand(n, 1)
    h = np.random.rand(m, 1)

    # Constant
    eps = 0.01
    TVe = TV / eps
    temporal_cost_e = temporal_cost / eps
    # Call df multiple times
    for w in np.linspace(0.1, 1.0, 50):
        result = df_test(w, TV, temporal_cost, g, h, eps)
        result2 = df_test_loop(w, TVe, temporal_cost_e, g, h, eps)
        #result3 = df_test_opt(w, TVe, temporal_cost_e, g, h, eps)
    print("Example output:", result)
    print(f'results2: {result2}')


def time_analyse2():
    np.random.seed(42)
    size = 80
    x = np.linspace(-1, 1, size)
    X1 = norm.pdf(x, -1, 0.5)  # First Gaussian
    X2 = norm.pdf(x, 1, 0.5)   # Second Gaussian

# Normalize to create proper distributions
    X1 = X1 / np.sum(X1)
    X2 = X2 / np.sum(X2)

    def eTiOTold2(X,Y, timing = None):
        return eTiOTold(X,Y, eps = 0.05, freq = 10,  verbose=2, timing=True, eta=5*10**-5, init_stepsize=False, subprob_tol=0.01 )
    def eTiOT2(X,Y, timing = None):
        return eTiOT(X,Y, eps = 0.05, freq = 10,  verbose=2, timing=True, eta=5*10**-5, init_stepsize=False, subprob_tol=0.01)
    def eTAOTold2(X,Y, timing = None):
        return eTAOTold(X,Y, eps = 0.05, freq = 1,  verbose=2, timing=True)
    def eTAOT2(X,Y, timing = None):
        return eTAOT(X,Y, eps = 0.05, freq = 1,  verbose=2, timing=True)
    metrics = [TiOTold, TiOT, eTiOTold2, eTiOT2, TAOTold, TAOT, eTAOTold2, eTAOT2]
    for metric in metrics:
        result = metric(X1, X2, timing = True)
        print(f"Results: {result[0]}")
        print(f"Metric {metric.__name__} has time consumed: {result[-1]}")
        print("\n ---------------------------------------------------------------- \n")


def time_analyse():
    # time_analyse1()
    time_analyse2()


# if __name__ == '__main__':
#     time_analyse()