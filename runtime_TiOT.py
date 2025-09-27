import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
#from TiOT_lib import TiOT,  TAOT, sinkhorn
import os
import time
import csv
import seaborn as sns
import cProfile
from scipy.stats import norm

def eTiOT(X,Y, verbose = False, eps = 0.1, freq = 10, eta =0.05,  subprob_tol = 0.01, init_stepsize = False, submax_iter = 50, solver = 'PGD', maxIter = 5000, tol = 0.005): #default for align data eta = 0.05
    return TiOT_lib.eTiOT(X,Y, eps = eps, freq = freq,  verbose=2, timing=True, eta=eta, init_stepsize=init_stepsize, subprob_tol=subprob_tol, submax_iter=submax_iter, solver=solver, tolerance=tol, maxIter=maxIter)

def eTAOT(X,Y, verbose = False):
    return TiOT_lib.eTAOT(X,Y, eps = 0.1, freq = 1,  verbose=2, timing=True)

def TiOT(X,Y):
    return TiOT_lib.TiOT(X,Y, timing=True, detail_mode=True)

def TAOT(X,Y):
    return TiOT_lib.TAOT(X,Y, timing=True, w = 0.5)


def plot_runtime(results, plot_file, logscale):
    lengths = results['len']
    metric_names = [k for k in results.keys() if k != 'len']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(9, 7))
    plt.grid(False)
    markers = ['s', 'o',  '^', 'D', 'v', 'P', 'X']
    i = 0
    colors1 = ['#6d36ab', '#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'] # '#1f77b4'
    colors2 = ["#07283e", "#582c05", "#0a460a", "#920e0e"] # '#1f77b4'
    name_dict = {'TiOT':   r'$\mathcal{D}_2$', 'eTiOT':  r'$\mathcal{D}_2^\varepsilon$', 'TAOT':   r'$\mathcal{W}_{2,0.5}$', 'eTAOT':  r'$\mathcal{W}_{2,0.5}^\varepsilon$'}
    for name in metric_names:
        plt.plot(lengths[:len(results[name])], results[name], label = name_dict[name], color = colors1[i], linewidth=1.75, marker = markers[i], markersize = 6)
        i+=1
    if logscale:
        plt.yscale("log")  
        #plt.xscale("log")  
    plt.tick_params(axis="both", which="major", labelsize=20, bottom=True, left=True)
    plt.xlabel("Series' lengths", fontsize = 20)
    plt.ylabel("Running time (s)", fontsize = 20)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        fontsize=20,
        frameon=False
    )

    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.25) 
    plt.savefig(plot_file, dpi=300)  # High-resolution
    plt.show()

def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def generate_data(size, seed=42, noise=None, l = -1 , r = 1):
    np.random.seed(seed)
    x = np.linspace(l, r, size)
    X1 = norm.pdf(x, l, 1)  # First Gaussian
    X2 = norm.pdf(x, r, 1)   # Second Gaussian

    if noise is not None:
        # Noise scales relative to value at each point
        X1 = X1 + np.random.normal(0, noise* X1 , size) #* X1
        X2 = X2 + np.random.normal(0, noise* X2 , size) # * X2
        # X1 = X1 + np.random.uniform(-noise, noise, size)
        # X2 = X2 + np.random.uniform(-noise, noise, size)
        # Prevent negatives if needed
        X1 = np.clip(X1, 0, None)
        X2 = np.clip(X2, 0, None)

    return X1 / np.sum(X1), X2 / np.sum(X2)

def gaussians(n, peaks, heights, widths, noise = 0.01, seed = 13):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.zeros_like(t, dtype=float)
    for p, h, w in zip(peaks, heights, widths):
        y += h * np.exp(-(t - p) ** 2 / (2 * w ** 2))
    y += rng.normal(0, noise, size=n)
    return y

def generate_two_gaussian(n, noise=0.01, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # --- Define relative peak positions (fractions of n) ---
    peak_positions = [0.5 * n, 0.9 * n]  

    # --- Define relative widths and heights (scale with n) ---
    base_widths = [0.01 * n, 0.02 * n]     
    base_heights = [1.2, 0.2]             

    # --- Create first series ---
    y1 = np.zeros_like(t, dtype=float)
    for p, h, w in zip(peak_positions, base_heights, base_widths):
        y1 += h * np.exp(-(t - p) ** 2 / (2 * w ** 2))
    y1 += rng.normal(0, noise, size=n)

    # --- Create second series: slightly shifted/scaled versions ---
    y2 = np.zeros_like(t, dtype=float)
    for p, h, w in zip(peak_positions, base_heights, base_widths):
        p_shift = p - 0.4 * n #+ rng.uniform(-0.1*n, 0.1*n)   # small random shift
        h_var   = h * rng.uniform(0.9, 1.1)          # ~10% variation in height
        w_var   = w * rng.uniform(0.9, 1.1)          # ~10% variation in width
        y2 += h_var * np.exp(-(t - p_shift) ** 2 / (2 * w_var ** 2))
    y2 += rng.normal(0, noise, size=n)

    return y1, y2

def runtime_experiment():
    RUN = False
    logscale = False
    if logscale:
        logscale_str = '_logscale'
    else:
        logscale_str = ""
    dataset_name = 'Gaussian'
    lengths = [200,400,600, 2000, 4000, 6000, 8000, 10000]
    metrics = [TiOT, TAOT, eTiOT,  eTAOT] #TiOT, TAOT,
    metric_names = [metric.__name__ for metric in metrics]
    results = {**{'len': lengths}, **{name: [] for name in metric_names}}
    result_file = os.path.join("runningtime_data", f"Results runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})" + 'seed6_final' + ".csv")
    plot_file = os.path.join("runningtime_data", f"Plot runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})" + logscale_str  + 'seed6_final' +  ".pdf")

    if RUN:
        for length in lengths:
            X, Y = generate_two_gaussian(n = length)
            #X, Y = generate_data(length)
            print(f"Start length {length}")
            for metric in metrics:
                if metric == TiOT and length > 800:
                    print(f"  ---> TiOT is skipped")
                    results[metric.__name__].append(None)
                    continue
                # if metric == eTiOT:
                #     d, plan, w, t = metric(X,Y)
                #     print(f'optimal w = {w}')
                print(f"  ---> Start metric {metric.__name__}")
                # if metric == TAOT:
                #     d, plan, t = metric(X,Y)
                #     nonzero_count = np.count_nonzero(plan)
                #     total_count = plan.size
                #     percentage = (nonzero_count / total_count) * 100
                #     print(f"Percentage of nonzero elements: {percentage:.2f}%")
                #     print(f"Plan 1st row is", plan[1])
                results[metric.__name__].append(metric(X,Y)[-1])
        save_result(results, result_file)
        plot_runtime(results, plot_file, logscale)
    else:
        results = read_result(result_file)
        plot_runtime(results, plot_file, logscale)

def deviation_experiment():
    size = 100
    noise = 1
    seeds = [i for i in range(100)]
    lamdas = [2, 10, 50, 100] #2, 10, 50, 100, 
    eps_list = 1/np.array(lamdas)
    plot_file = os.path.join("runningtime_data", 'boxplot' + f"_n = {size}__noise = {noise}_seed = {len(seeds)}_"  + ".pdf")
    result_file = os.path.join("runningtime_data", 'boxplot' + f"_n = {size}__noise = {noise}__seed = {len(seeds)}_" + ".csv")
    result_w_file = os.path.join("runningtime_data", 'boxplot' + f"_n = {size}__noise = {noise}_seed = {len(seeds)}__" + ".csv")
    results = dict()
    results_w = dict()
    RUN = False
    if RUN :
        for eps in eps_list:
            deviation = []
            w_deviation = []
            print(f"Start experimenting with eps = {eps}")
            for seed in seeds:
                #X, Y = generate_data(size=size, seed=seed, noise=noise, l = -1.5, r = 1.5)
                X, Y = generate_two_gaussian(n = size, seed=seed)
                emd, plan_emd,  w_emd, t = TiOT(X,Y)
                sinkhorn, plan_sink, w_sink,t  = eTiOT(X,Y, eps=eps, eta = 0.002, subprob_tol=10**-7, submax_iter=400, tol = 10**-9, maxIter = 50000)
                print(f"At seed {seed}: emd = {emd} with w = {w_emd}, sinkhorn = {sinkhorn} with w = {w_sink}")
                deviation.append(np.abs(sinkhorn - emd) / emd)
                w_deviation.append(np.abs(w_emd - w_sink) / w_emd )
            results[eps] = deviation
            results_w[eps] = w_deviation
    else:
        results = read_result(result_file)
        results_w = read_result(result_w_file)

    save_result(results, result_file)
    save_result(results_w, result_w_file)

    positions = np.arange(1, len(lamdas) + 1)
    width = 0.3

    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(False)
    # Red outline, no fill (results)
    bp1 = ax.boxplot(list(results.values()), positions=positions - width/2, widths=0.25,
                    patch_artist=False,
                    boxprops=dict(color='#B22222', linewidth=1.2),
                    capprops=dict(color='#B22222', linewidth=1.2),
                    whiskerprops=dict(color='#B22222', linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor='#B22222', markeredgecolor='#B22222'),
                    medianprops=dict(color='#B22222', linewidth=1.2))

    # Gray outline, dashed style (results_w)
    bp2 = ax.boxplot(list(results_w.values()), positions=positions + width/2, widths=0.25,
                    patch_artist=False,
                    boxprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    capprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    whiskerprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    flierprops=dict(marker='s', markersize=3, markerfacecolor='#008080', markeredgecolor='#008080'),
                    medianprops=dict(color='#008080', linewidth=1.2))

    # Axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(lamdas, fontsize=16)
    # ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=21, bottom=True, left=True)
    ax.set_ylabel('Distribution of deviation', fontsize=20, labelpad=12)
    ax.set_xlabel(r'$1/\varepsilon $', fontsize=20, labelpad=12)
    # Legend (clean style)
    # ax.legend([bp2["boxes"][0], bp1["boxes"][0]], [ r"$\frac{|w^*_{\varepsilon} - w^*|}{w^*}$", r"$\frac{| \langle C(w^*_{\varepsilon}), \pi^*_{\varepsilon} \rangle - \langle C(w^*), \pi^* \rangle |}{\langle C(w^*), \pi^* \rangle}$"],
    #         loc="best", fontsize=22, frameon=True)
    ax.legend(
    [bp2["boxes"][0], bp1["boxes"][0]],
    [
        r"$\frac{|w^*_{\varepsilon} - w^*|}{w^*}$",
        r"$\frac{| \langle C(w^*_{\varepsilon}), \pi^*_{\varepsilon} \rangle - \langle C(w^*), \pi^* \rangle |}{\langle C(w^*), \pi^* \rangle}$"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),  # push further down
    ncol=2,
    fontsize=22,
    frameon=False                  # remove legend box
)
    plt.yscale("log") 
    # Tight layout
    plt.tight_layout()
    plt.savefig(plot_file , dpi=300)
    plt.show()


def main():
    runtime_experiment()

main()