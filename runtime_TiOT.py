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

def eTiOT(X,Y, verbose = False, timing  = True):
    return TiOT_lib.eTiOT(X,Y, eps = 0.05, freq = 10,  verbose=2, timing=True, eta=5*10**-5, init_stepsize=False, subprob_tol=0.01)

def eTAOT(X,Y, verbose = False, timing  = True):
    return TiOT_lib.eTAOT(X,Y, eps = 0.05, freq = 1,  verbose=2, timing=True)

def TiOT(X,Y):
    return TiOT_lib.TiOT(X,Y, timing=True)

def TAOT(X,Y):
    return TiOT_lib.TAOT(X,Y, timing=True)

def sinkhorn(x,y, verbose = False, timing  = True):
    return TiOT_lib.sinkhorn(x,y, eps=0.05)

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
            if metric == TiOT  and length > 1000:
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
    colors1 = [['#6d36ab','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']] # '#1f77b4'
    colors2 = ["#07283e", "#582c05", "#0a460a", "#920e0e"] # '#1f77b4'

    for name in metric_names:
        plt.plot(lengths[:len(results[name])], results[name], label = name, color = colors1[i], linewidth=1.75, marker = markers[i])
        i+=1
    if logscale:
        plt.yscale("log")  
        plt.xscale("log")  
    plt.xlabel("Series' lengths", fontsize = 14)
    plt.ylabel("Running time (s)", fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)  # High-resolution
    plt.show()

def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def generate_data(size):
    np.random.seed(42)
    x = np.linspace(-1, 1, size)
    X1 = norm.pdf(x, -1, 0.5)  # First Gaussian
    X2 = norm.pdf(x, 1, 0.5)   # Second Gaussian
    return X1 / np.sum(X1), X2 / np.sum(X2)

def main():
    RUN = True
    logscale = False
    if logscale:
        logscale_str = '_logscale'
    else:
        logscale_str = ""
    dataset_name = 'Gaussian'
    lengths = [ 200, 400, 800, 2000]
    metrics = [ TiOT, TAOT, eTiOT,  eTAOT] #TiOT, TAOT,
    metric_names = [metric.__name__ for metric in metrics]
    results = {**{'len': lengths}, **{name: [] for name in metric_names}}
    result_file = os.path.join("runningtime_data", f"Results runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})"  + ".csv")
    plot_file = os.path.join("runningtime_data", f"Plot runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})" + logscale_str + ".pdf")

    if RUN:
        for length in lengths:
            X, Y = generate_data(length)
            print(f"Start length {length}")
            for metric in metrics:
                if metric == TiOT and length > 1000:
                    print(f"  ---> TiOT is skipped")
                    results[metric.__name__].append(None)
                    continue
                print(f"  ---> Start metric {metric.__name__}")
                results[metric.__name__].append(metric(X,Y)[-1])
        save_result(results, result_file)
        plot_runtime(results, plot_file, logscale)
    else:
        results = read_result(result_file)
        plot_runtime(results, plot_file, logscale)

main()