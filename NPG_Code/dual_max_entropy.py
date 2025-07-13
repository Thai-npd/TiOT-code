import numpy as np
from save_and_plot import  make_all_plots, save, load
from alg import AdProxGrad, ProxGrad, NPG, AdaPG


# Control section
# - If data is not loaded then it will be created.
# - If results is not loaded then the algorithms will run.
######################################################
LOAD_DATA = False
SAVE_DATA = False
LOAD_RESULTS = False
SAVE_RESULTS = True
PLOT = True
######################################################

seed = 4
m, n = 500,2000
N = 200

def run_dual_max_entropy(m = m, n = n, seed = seed, N = N):
    name = "dual_max_ent"
    name_instance = f"{name}_m={m}_n={n}_normal_seed={seed}"
    np.random.seed(seed)
    y0 = np.zeros(m + 1)
    tol = 1e-6

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
    else:
        A = np.random.randn(m, n)
        x_feas = np.random.uniform(0.1, 1, n)
        x_feas /= np.linalg.norm(x_feas, ord=1)
        b = A @ x_feas

    if SAVE_DATA:
        data = {"A":A, "b":b}
        save(data, name_instance, name,saving_obj=1)

    def f_dual(y):
        la = y[:-1]
        nu = y[-1]
        return np.dot(b, la) + nu + np.sum(np.exp(-nu - 1) * np.exp(-A.T @ la))

    def df_dual(y):
        la = y[:-1]
        nu = y[-1]
        df_la = -A @ (np.exp(-nu - 1) * np.exp(-A.T @ la)) + b
        df_nu = 1 - np.sum(np.exp(-nu - 1) * np.exp(-A.T @ la))
        return np.append(df_la,df_nu)

    def oracle_f(y):
        return f_dual(y), df_dual(y)

    def g(y):
        return 0

    def prox_g(y, alpha):
        if np.ndim(y) == 2: 
            print(y)
            print(y.shape)
            exit()
        y[:-1] = np.maximum(0, y[:-1])
        return y

    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, y0, maxit = N, tol = tol, stop = "res", lns_init = False, verbose = False, 
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
    parameters = {"additional_info":"" , "size":f"({m}, {n})", "seed": seed}
    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_dual_max_entropy()