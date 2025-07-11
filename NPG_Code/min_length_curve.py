import numpy as np
from save_and_plot import load, save, make_all_plots
from alg import AdProxGrad, ProxGrad, NPG, AdaPG

# Control section
# - If data is not loaded then it will be created.
# - If results is not loaded then the algorithms will run.
######################################################
LOAD_DATA = False
SAVE_DATA = False
LOAD_RESULTS = False
SAVE_RESULTS = True
PLOT = False
######################################################

seed = 1
m, n = 500,5000
N = 50000


def run_min_len_curve(m = m, n = n, seed = seed, N = N):
    def f(x):
        return np.sqrt(1 + x[0]**2) + np.sum(np.sqrt(1 + (x[1:] - x[:-1])**2))

    def df(x):
        elements = np.insert( np.sqrt(1 + (x[1:] - x[:-1])**2),0,np.sqrt(1 + x[0]**2))
        return -np.diff(np.append(np.insert(np.diff(x),0,x[0]) * (1 / elements),0))

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return 0.0

    def prox_g(x, alpha):
        return x - A.T.dot(P.dot(A.dot(x) - b))

    name = "min_len_curve"
    np.random.seed(seed)
    name_instance = f"{name}_m={m}_n={n}_seed={seed}"
    tol = 1e-6

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
        P  = loaded_data["P"]
        x_feas = loaded_data["x_feas"]
        x0 = loaded_data["x0"]
    else:
        x_feas = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A.dot(x_feas)
        P = np.linalg.inv(A.dot(A.T))
        x0 = prox_g(np.random.randn(n), 1)

    if SAVE_DATA:
        data = {"A":A, "b":b, "x0":x0,"P":P,"x_feas":x_feas}
        save(data, name_instance, name,saving_obj=1)

    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = True, verbose = False, 
                                 fixed_step = 0.001, **kwargs)
    
    results = {}
    if LOAD_RESULTS:
        loaded_results = load(name_instance, name, saving_obj=2)
        for key, history in loaded_results.items():
            results[key] = history
    else:          
        x1, history1 = Run(NPG, params = [0.1, 5.7, 0.7, 0.69], ver=1)
        results["NPG1"] = history1

        x2, history2 = Run(NPG, params = [0.1, 5.7, 0.99, 0.98], ver=2)
        results["NPG2"] = history2
  
        x3, history3 = Run(AdProxGrad)
        results["AdPG"] = history3
        
        x4, history4 = Run(AdaPG, params=[3/2, 3/4])
        results["AdaPG" + str([3/2, 3/4])] = history4

        for param in [[1.1, 0.5], [1.2, 0.5]]:
            x5, history5 = Run(ProxGrad, params = param)
            results["PG-LS" + str(param)] = history5


    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    make_all_plots(results, name_instance, name, plot=PLOT)
    parameters = {"additional_info": "", "size":f"({m}, {n})", "seed": seed}
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_min_len_curve()