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


def plot_graph(results, plot_file):
    lags = results['lag']
    metric_names = [k for k in results.keys() if k != 'lag']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['', 'o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':', '-', '-', '-', '-']
    i = 0
    for name in metric_names:
        plt.plot(lags, results[name], label = name, linewidth=1.75, marker = markers[i], linestyle = linestyles[i])
        i+=1
    plt.xlabel("Lag (days)", fontsize = 14)
    plt.ylabel("Distance", fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)  # High-resolution
    plt.show()

def save_result(results, result_file):
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())  # header
        rows = zip(*results.values())    # transpose
        writer.writerows(rows)  

def read_result(result_file):
    results = {}
    with open(result_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        alg_names = header[1:]
        results['lag'] = []
        for name in alg_names:
            results[name] = []
        for row in reader:
            results['lag'].append(row[0])
            for i, name in enumerate(alg_names):
                if row[i+1] != '':
                    results[name].append(float(row[i + 1]))
                else:
                    results[name].append(None)

def main():
    RUN = True
    lags = range(0, 730)
    length = 365
    file_path = 'DailyDelhiClimateTrain.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    w_list = [0.25, 0.5, 0.75]
    metric_names = ['TiOT'] + [f"TAOT(w = {w})" for w in w_list]
    results = {**{'lag' : lags}, **{name: [] for name in metric_names}}

    result_file = os.path.join("lag_series_data", f"Results distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]}).csv")
    plot_file = os.path.join("lag_series_data", f"Plot distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]}).pdf")

    if RUN:
        for lag in lags:
            results['TiOT'].append(TiOT(df['meantemp'].iloc[:length], df['meantemp'].iloc[lag:lag+length])[0])
            for w in w_list:
                results[f'TAOT(w = {w})'].append(TAOT(df['meantemp'].iloc[:length], df['meantemp'].iloc[lag:lag+length], w = w)[0])
            print(f"Done Lag = {lag}")
        save_result(results, result_file)
        plot_graph(results, plot_file)
    else:
        results = read_result(result_file)
        plot_graph(results, plot_file)

main()