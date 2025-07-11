import numpy as np
from save_and_plot import save, load, make_all_plots
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
m, r, n = 500, 20, 1000


def run_nmf(m = m, r = r, n = n, seed = seed):
    def f(X):
        U, V = X[:m], X[m:]
        return 0.5 * np.linalg.norm(np.dot(U, V.T) - A)**2

    def oracle_f(X):
        U, V = X[:m], X[m:]
        res = U @ V.T - A
        grad_U = res @ V
        grad_V = res.T @ U
        return 0.5 * np.linalg.norm(res)**2, np.vstack([grad_U, grad_V])

    def g(x):
        return 0.0

    def prox_g(x, alpha):
        return np.vstack([np.maximum(x[:m], 0.0), np.maximum(x[m:], 0.0)])


    np.random.seed(seed)
    name = "nonnegative_matrix_factorization"
    name_instance = f"nmf_m={m}_n={n}_r={r}_seed={seed}"
    tol = 1e-6
    N = 2000

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        x0 = loaded_data["x0"]
    else:
        B = np.maximum(np.random.randn(m, r), 0)
        C = np.maximum(np.random.randn(n, r), 0)
        A = np.dot(B, C.T)
        U0 = np.random.rand(m, r)
        V0 = np.random.rand(n, r)
        x0 = np.vstack([U0, V0])
    if SAVE_DATA:
        data = {"A":A, "x0":x0}
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
    parameters = {"additional_info": "", "size":f"({m}, {r}, {n})", "seed": seed}
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_nmf()