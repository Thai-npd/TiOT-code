import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
from TiOT_lib import TiOT,  TAOT, sinkhorn
import os
import time
import csv
import seaborn as sns
import cProfile
def process_data(dataset_name, start1, start2, numpoint ):
    filepath = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TRAIN.txt" )

    with open(filepath, "r") as file:
        data = [line.strip().split() for line in file]

    # Convert to numerical values if needed
    data = [[float(value) for value in row] for row in data]

    X = [row[1:] for row in data]
    X1 = X[start1: start1 + numpoint]
    X2 = X[start2: start2 + numpoint]
    return X1, X2

def eTiOT(x,y, verbose = False, timing  = True):
    return TiOT_lib.eTiOT(x, y, eps=0.1, freq=10, eta=1.5,  verbose=verbose, timing=timing, init_stepsize=False, subprob_tol=0.01)

def eTAOT(x,y, verbose = False, timing  = True):
    return TiOT_lib.eTAOT(x, y, eps=0.1, freq=10, verbose=verbose, timing=timing)

def sinkhorn(x,y, verbose = False, timing  = True):
    return TiOT_lib.sinkhorn(x,y, eps=0.1)

def get_runtime(x,y, metric):
    outputs = metric(x,y, verbose = True, timing = True)
    return outputs[-1]

def combine_runtimes(X1, X2, metrics, lengths):
    num_point = len(X1)
    metric_names = [metric.__name__ for metric in metrics]
    results = {**{'len': lengths}, **{name: [] for name in metric_names}}
    for length in lengths:
        print(f"Start length {length}")
        for metric in metrics:
            if metric == TiOT  and length > 600:
                results[metric.__name__].append(None)
                print(f"  ===> Done algorithm {metric.__name__} ")
                continue
            s = 0
            for i in range(num_point):
                s += get_runtime(X1[i][:length], X2[i][:length], metric)
            results[metric.__name__].append(s / num_point)
            print(f"  ===> Done algorithm {metric.__name__} ")
    return results

def plot_runtime(results, plot_file, logscale):
    lengths = results['len']
    metric_names = [k for k in results.keys() if k != 'len']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    i = 0
    for name in metric_names:
        plt.plot(lengths[:len(results[name])], results[name], label = name, linewidth=1.75, marker = markers[i])
        i+=1
    if logscale:
        plt.yscale("log")  
        plt.xscale("log")  
    plt.xlabel("Series' lengths", fontsize = 14)
    plt.ylabel("Running time (s)", fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)  # High-resolution
    #plt.show()

def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def gaussian_mixture_timeseries(length, n_components=3, weights=None, means=None, stds=None, random_state=None):
    """
    Generate a Gaussian mixture time series.
    
    Parameters:
        length (int): Length of the time series.
        n_components (int): Number of Gaussian components.
        weights (list or None): Mixing weights (must sum to 1). If None, uniform weights are used.
        means (list or None): Means of Gaussians. If None, random values are used.
        stds (list or None): Standard deviations of Gaussians. If None, random values are used.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        ts (ndarray): Generated time series of shape (length,).
        component_ids (ndarray): The component index chosen at each time step.
    """
    rng = np.random.default_rng(random_state)

    # Default weights
    if weights is None:
        weights = np.ones(n_components) / n_components
    weights = np.array(weights) / np.sum(weights)  # normalize
    
    # Default means and stds
    if means is None:
        means = rng.uniform(-5, 5, size=n_components)
    if stds is None:
        stds = rng.uniform(0.5, 2.0, size=n_components)

    # Sample component for each time step
    component_ids = rng.choice(n_components, size=length, p=weights)
    
    # Draw samples from corresponding Gaussians
    ts = rng.normal(means[component_ids], stds[component_ids])
    
    return ts

def main():
    RUN = True
    logscale = False
    if logscale:
        logscale_str = '_logscale'
    else:
        logscale_str = ""
    dataset_name = 'Gaussian' # PigCVP, Rock
    #lengths = [100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1500, 1800, 2100] #100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1500, 1800, 2100, 2400, 2800
    lengths = [ 2000] #
    metrics = [ eTiOT,  eTAOT] #TiOT, TAOT,
    result_file = os.path.join("runningtime_data", f"Results runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})"  + ".csv")
    plot_file = os.path.join("runningtime_data", f"Plot runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})" + logscale_str + ".pdf")

    if RUN:
        #X1, X2 = process_data(dataset_name, start1=0, start2=10, numpoint=1)
        X1  = gaussian_mixture_timeseries(10000, n_components=200, random_state=0).reshape(1,-1)
        X2 = gaussian_mixture_timeseries(10000, n_components=200, random_state=1).reshape(1,-1)
        results = combine_runtimes(X1, X2, metrics, lengths)
        save_result(results, result_file)
        plot_runtime(results, plot_file, logscale)
    else:
        results = read_result(result_file)
        plot_runtime(results, plot_file, logscale)

def time_analyse():
    profiler = cProfile.Profile()
    lengths = [2000]
    X1  = gaussian_mixture_timeseries(2000, n_components=200, random_state=0)
    X2 = gaussian_mixture_timeseries(2000, n_components=200, random_state=1)
    profiler.enable()
    metric = eTiOT
    print(X1.shape)
    results =  metric(X1, X2)
    print(results)
    profiler.disable()
    profiler.dump_stats(f"{metric.__name__}output.prof")
    print("Profiling finished. Results saved to output.prof")    



time_analyse()