import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
from TiOT_lib import TiOT, eTiOT, TAOT, eTAOT
import os
import time
import csv
import seaborn as sns

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

def fast_eTiOT(x,y, verbose = False):
    return eTiOT(x, y, w_update_freq=20, verbose=verbose)

def get_runtime(x,y, metric):
    start = time.perf_counter()
    metric(x,y, verbose = True)
    end = time.perf_counter()
    return end - start

def combine_runtimes(X1, X2, metrics, lengths):
    num_point = len(X1)
    metric_names = [metric.__name__ for metric in metrics]
    results = {**{'len': lengths}, **{name: [] for name in metric_names}}
    for length in lengths:
        print(f"Start length {length}")
        for metric in metrics:
            if metric == TiOT  and length > 900:
                results[metric.__name__].append(None)
                print(f"  ===> Done algorithm {metric.__name__} ")
                continue
            s = 0
            for i in range(num_point):
                s += get_runtime(X1[i][:length], X2[i][:length], metric)
            results[metric.__name__].append(s / num_point)
            print(f"  ===> Done algorithm {metric.__name__} ")
    return results

def plot_runtime(results, plot_file):
    lengths = results['len']
    metric_names = [k for k in results.keys() if k != 'len']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    i = 0
    for name in metric_names:
        plt.plot(lengths[:len(results[name])], results[name], label = name, linewidth=1.75, marker = markers[i])
        i+=1
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

def main():
    RUN = True
    dataset_name = 'PigCVP' # PigCVP Rock
    #lengths = [100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1500, 1800, 2100] #100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1500, 1800, 2100, 2400, 2800
    lengths = [100, 200, 300, 400, 600, 800, 1000, 1300, 1600, 2000] #100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1500, 1800, 2100, 2400, 2800
    metrics = [TiOT, eTiOT, fast_eTiOT, eTAOT]
    result_file = os.path.join("runningtime_data", f"Results runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]}).csv")
    plot_file = os.path.join("runningtime_data", f"Plot runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]}).pdf")

    if RUN:
        X1, X2 = process_data(dataset_name, start1=0, start2=10, numpoint=3)
        results = combine_runtimes(X1, X2, metrics, lengths)
        save_result(results, result_file)
        plot_runtime(results, plot_file)
    else:
        results = read_result(result_file)
        plot_runtime(results, plot_file)


main()