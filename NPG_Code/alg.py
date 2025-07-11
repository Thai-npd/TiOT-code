import numpy as np
import time
import warnings

# Suppress RuntimeWarning: divide by zero
warnings.filterwarnings("ignore", category=RuntimeWarning)


def mse(x1,x2):
    return np.mean((x1 - x2)**2)

def linesearch_initial(oracle_f, g, prox_g, x0, t, incr = 10, dcr = 0.5, veryLargeSize = 10):
    obj_x, grad_fx0 = oracle_f(x0)
    largestep = True
    for i in range(1, 101):
        x1 = prox_g(x0 - t * grad_fx0, t)
        obj_x1, grad_fx1 = oracle_f(x1)
        if i == 1 and np.linalg.norm(x0 - x1) < 10**-6: 
            print("Congrats: initial x0 is a solution")
        else:
            L = np.linalg.norm(grad_fx1 - grad_fx0) / np.linalg.norm(x1 - x0)
            if t * L > 2:
                largestep = False
                t *= dcr
            else:
                if t * L <= 2 and largestep:
                    t *= incr
                    if t > veryLargeSize:
                        return t, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                else:
                    return t, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i

def linesearch(oracle_f, g, prox_g, x, fx, grad_fx, t, inc=1.2, dcr=0.5):
    t *= inc
    for i in range(1, 1001):
        x1 = prox_g(x - t * grad_fx, t)
        if np.linalg.norm(x1 - x) < 10**-20 or t < 10**-20:
            return x, fx, grad_fx, t
        fx1, grad_fx1 = oracle_f(x1)
        if fx1 <= fx + np.sum(grad_fx * (x1 - x))  + 1.0 / (2 * t) * np.linalg.norm(x1 - x)**2: #np.dot(grad_fx, x1 - x)
            x, fx, grad_fx = x1, fx1, grad_fx1
            break
        else:
            t *= dcr
    return x, fx, grad_fx, t

def explicit_linesearch1(oracle_f,g, prox_g, x, fx, grad_fx, t = 1, inc=1.2, dcr=0.5):
    t *= inc
    t = 0.1
    delta = 0.45
    for i in range(1, 1001):
        x1 = prox_g(x - t * grad_fx, t)
        fx1, grad_fx1 = oracle_f(x1)
        if np.linalg.norm(grad_fx1 - grad_fx) / np.linalg.norm(x1 - x) <= delta / t:
            x, fx, grad_fx = x1, fx1, grad_fx1
            break
        else:
            t *= dcr
    return x, fx, grad_fx, t,

def explicit_linesearch2(oracle_f,g, prox_g, x, fx, grad_fx, t = 1, inc=1.2, dcr=0.5):
    t *= inc
    t = 1
    y = prox_g(x - 1 * grad_fx, 1)
    fix_term = g(y) - g(x) + np.dot(grad_fx, y - x) + (1/2)* np.linalg.norm(y-x)**2
    for i in range(1, 1001):
        x1 = x + t * (y - x)
        fx1, grad_fx1 = oracle_f(x1)
        if fx1 <= fx + t * fix_term:
            x, fx, grad_fx = x1, fx1, grad_fx1
            break
        else:
            t *= dcr
    return x, fx, grad_fx, t,

def collect_history(history_dic, data_dic):
    for key in history_dic.keys():
        history_dic[key].append(data_dic[key])
    return history_dic

def AdProxGrad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2,params =None, xopt = 0, ver=2):
    def obj(fx, x):
        return fx + g(x)
    start = time.perf_counter()
    
    x_prev = x0
    theta = 1.0 / 3
    
    if lns_init:
        t_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        if verbose:
            print(f"Linesearch found initial stepsize {t_prev} in {lns_iter} iterations")
    else:
        t_prev = fixed_step
        if verbose:
            print("No linesearch, initial stepsize is ", t_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - t_prev * grad_prev, t_prev)
        fx, grad_fx = oracle_f(x)
    dict_values = {
        "res": [ np.linalg.norm(x - x_prev) / t_prev], 
        "obj": [ obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [ t_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}
    for i in range(1, maxit):
        if ver == 1:
            L = np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev)
            t = min(np.sqrt(1 + theta) * t_prev, 1 / (np.sqrt(2) * L))
        elif ver == 2:
            L = np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev)
            t = min(np.sqrt(2 / 3 + theta) * t_prev, t_prev / np.sqrt(max(2 * t_prev**2 * L**2 - 1, 0)))
        elif ver == 3:
            t = fixed_step
        theta = t / t_prev
        x_prev, grad_prev, t_prev = x, grad_fx, t
        x = prox_g(x - t * grad_fx, t)
        end = time.perf_counter()

        residual = np.linalg.norm(x_prev - x) / t
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": t,"time":end-start, "mse":mse(x, xopt)}
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The AdPG algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The AdPG algorithm terminated without reaching required accuracy")
    return x, history_dic

def AdaPG(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2,params =None, xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    start = time.perf_counter()
    
    x_prev = x0
    theta = 1.0
    
    if lns_init:
        t_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        if verbose:
            print(f"Linesearch found initial stepsize {t_prev} in {lns_iter} iterations")
    else:
        t_prev = fixed_step
        if verbose:
            print("No linesearch, initial stepsize is ", t_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - t_prev * grad_prev, t_prev)
        fx, grad_fx = oracle_f(x)
    dict_values = {
        "res": [ np.linalg.norm(x - x_prev) / t_prev], 
        "obj": [ obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [ t_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}
    q = params[0]
    r = params[1]
    for i in range(1, maxit):
        L = np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev)
        l = np.sum( (grad_fx - grad_prev)*(x - x_prev) )  / np.linalg.norm(x - x_prev)**2
        t = t_prev * min(np.sqrt(1/q + theta), np.sqrt(1- r/q) / np.sqrt(max(t_prev**2 * L**2 + 2*t_prev*l*(r-1) - (2*r-1), 0) ))

        theta = t / t_prev
        x_prev, grad_prev, t_prev = x, grad_fx, t
        x = prox_g(x - t * grad_fx, t)
        end = time.perf_counter()

        residual = np.linalg.norm(x_prev - x) / t
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": t,"time":end-start, "mse":mse(x, xopt)}
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The AdaPG{params} algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The AdaPG{params} algorithm terminated without reaching required accuracy")
    return x, history_dic

def NPG(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, params = [0.1, 5.7, 0.7, 0.69], xopt = 0, ver=2):
    def obj(fx, x):
        return fx + g(x)
    def gamma_sequence(k):
        a = params[0]
        b = params[1]
        return (a * np.log(k+1)**b ) / (k+1)**1.1
    
    start = time.perf_counter()
    c0 = params[2]
    c1 = params[3]
    x_prev = x0

    if lns_init  :
        t_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=10) # try veryLargeSize = 10 if bad experiment results come
        t_prev_prev = t_prev
        if verbose:
            print(f"Linesearch found initial stepsize {t_prev} in {lns_iter} iterations")
    else:
        t_prev = t_prev_prev = fixed_step
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - t_prev * grad_prev, t_prev)
        fx, grad_fx = oracle_f(x)
        if verbose:
            print("No linesearch, initial stepsize is ", t_prev)
    dict_values = {
        "res": [np.linalg.norm(x - x_prev) / t_prev],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [t_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}

    for i in range(1, maxit):
        L =  np.linalg.norm(grad_fx - grad_prev) / np.linalg.norm(x - x_prev) 
        if L > c0 / t_prev :
            t = c1 / L
        else:
            gamma_prime = gamma_sequence(i-1)
            if t_prev / t_prev_prev < 1:
                gamma_prime  = np.min([gamma_prime, np.sqrt(1 + t_prev / t_prev_prev)-1])
            t = (1 + gamma_prime) * t_prev

        x_prev, grad_prev, t_prev,t_prev_prev = x, grad_fx, t, t_prev_prev
        x = prox_g(x - t * grad_fx, t)
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x) / t
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": t, "time":end-start, "mse":mse(x, xopt)}
                
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The NPG{ver} algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The NPG{ver} algorithm terminated without reaching required accuracy")    
    return x, history_dic

def NPG_quad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, params = [0.1,5.7,0.99,0.98], xopt = 0):
    def obj(fx, x):
        return fx + g(x)
    def gamma_sequence(k):
        a = params[0]
        b = params[1]
        return (a * np.log(k+1)**b ) / (k+1)**1.1
    
    start = time.perf_counter()
    c0 = params[2]
    c1 = params[3]
    x_prev = x0

    if lns_init  :
        t_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        t_prev_prev = t_prev
        if verbose:
            print(f"Linesearch found initial stepsize {t_prev} in {lns_iter} iterations")
    else:
        t_prev = t_prev_prev = fixed_step
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - t_prev * grad_prev, t_prev)
        fx, grad_fx = oracle_f(x)
        if verbose:
            print("No linesearch, initial stepsize is ", t_prev)
    dict_values = {
        "res": [np.linalg.norm(x - x_prev) / t_prev],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [t_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}

    for i in range(1, maxit):
        L = np.sum((x - x_prev).T @ (grad_fx - grad_prev)) / np.linalg.norm(x - x_prev)**2 
        if L > c0/t_prev :
            t = c1 / L
        else:
            gamma_prime = gamma_sequence(i-1)
            if t_prev / t_prev_prev < 1:
                gamma_prime  = np.min([gamma_prime, np.sqrt(1 + t_prev / t_prev_prev)-1])
            t = (1 + gamma_prime) * t_prev

        x_prev, grad_prev, t_prev,t_prev_prev = x, grad_fx, t, t_prev_prev
        x = prox_g(x - t * grad_fx, t)
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x) / t
        fx, grad_fx = oracle_f(x)
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": t, "time":end-start, "mse":mse(x, xopt)}
                
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The NPG-quad algorithm reached required accuracy in {i+1} iterations")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The NPG-quad algorithm terminated without reaching required accuracy")    
    return x, history_dic

def ProxGrad(oracle_f, g, prox_g, x0, maxit=1000, tol=1e-9, stop="res", lns_init=True, verbose=False, track=["res", "obj", "grad", "steps","time"], fixed_step=1e-2, params = [1.2,0.5], xopt = 0, LStype = 'armijo'):
    """
    LStype could only be either 'armijo' or 'explicit_LS1' or 'explicit_LS2'
    """
    def obj(fx, x):
        return fx + g(x)
    
    start = time.perf_counter()
    inc, dcr = params[0],params[1]
    x_prev = x0
    if LStype == 'explicit_LS1':
        alg_name = 'PG-eLS1'
        LS = explicit_linesearch1
    elif LStype == 'explicit_LS2':
        alg_name = 'PG-eLS2'
        LS = explicit_linesearch2
    else:
        alg_name = 'PG-LS'
        LS = linesearch        

    if lns_init:
        t_prev, x, grad_prev, grad_fx, f_prev, fx, lns_iter = linesearch_initial(oracle_f, g, prox_g, x0, fixed_step, veryLargeSize=0.9)
        if verbose:
            print(f"Linesearch found initial stepsize {t_prev} in {lns_iter} iterations")
    else:
        t_prev = fixed_step
        if verbose:
            print("No linesearch, initial stepsize is ", t_prev)
        f_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - t_prev * grad_prev, t_prev)
        fx, grad_fx = oracle_f(x)
    
    dict_values = {
        "res": [np.linalg.norm(x - x_prev) / t_prev],
        "obj": [obj(fx, x)],
        "grad": [np.linalg.norm(grad_fx)],
        "steps": [t_prev],
        "time":[0],
        "mse":[mse(x, xopt)]
    }
    history_dic = {key: [] for key in track}
    history_dic = {k: v for k, v in dict_values.items() if k in track}
    i = 0
    for _ in range(1, maxit):
        i += 1

        x, fx, grad_fx, t = LS(oracle_f, g,  prox_g, x, fx, grad_fx, t_prev,  inc=inc, dcr=dcr)
        end = time.perf_counter()
        residual = np.linalg.norm(x_prev - x) / t
        x_prev, t_prev = x, t
        current_info = {"res": residual, "obj": obj(fx, x), "grad": np.linalg.norm(grad_fx), "steps": t,"time":end-start, "mse":mse(x, xopt)}
        history_dic = collect_history(history_dic, current_info)
        if current_info[stop] <= tol:
            if verbose:
                print(f"The {alg_name}{params} algorithm reached required accuracy in {i+1} iterations.")
            break
    if current_info[stop] > tol:
        if verbose:
            print(f"The {alg_name}{params} algorithm terminated without reaching required accuracy")
    return x, history_dic