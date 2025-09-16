import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
from TiOT_lib import TiOT, eTiOT, TAOT, eTAOT
import os
import time
import csv
import seaborn as sns


def plot_graph(results, plot_file, index_name, x_label, y_label):
    indices = results[index_name]
    names = [k for k in results.keys() if k != index_name]
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['', 'o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':', '-', '-', '-', '-']
    i = 0
    for name in names:
        plt.plot(indices, results[name], label = name, linewidth=1.75, linestyle = linestyles[i])
        i+=1
    plt.xlabel(x_label, fontsize = 14)
    plt.ylabel(y_label, fontsize = 14)
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

def dist_lag_exp(RUN = True):
    start = 0
    lags = range(0, 730)
    length = 365
    file_path = 'DailyDelhiClimateTrain.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    w_list = [0.2, 0.5, 0.8]
    metric_names = ['TiOT'] + [f"TAOT(w = {w})" for w in w_list]
    results = {**{'lag' : lags}, **{name: [] for name in metric_names}}

    result_file = os.path.join("lag_series_data", f"Results distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]}).csv")
    plot_file = os.path.join("lag_series_data", f"Plot distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]}).pdf")

    if RUN:
        for lag in lags:
            results['TiOT'].append(TiOT(df['meantemp'].iloc[start:start + length], df['meantemp'].iloc[start + lag:start + lag+length])[0])
            for w in w_list:
                results[f'TAOT(w = {w})'].append(TAOT(df['meantemp'].iloc[start:start + length], df['meantemp'].iloc[start + lag:start + lag+length], w = w)[0])
            print(f"Done Lag = {lag}")
        save_result(results, result_file)
        plot_graph(results, plot_file, 'lag', 'Lag(days)', 'Distance')
    else:
        results = read_result(result_file)
        plot_graph(results, plot_file, 'lag', 'Lag(days)', 'Distance')

def dist_w_exp(RUN = True):
    file_path = 'DailyDelhiClimateTrain.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    lags = [20,90,180,270]
    w_list = [0.1 * i for i in range(11)]
    x = [df['meantemp'].iloc[:365]]
    dists = []
    result_file = os.path.join("lag_series_data", f"Results TAOT distances with w on {file_path}(lag {lags[0]} to {lags[-1]}).csv")
    plot_file = os.path.join("lag_series_data", f"Plot TAOT distances with w on {file_path}(lag {lags[0]} to {lags[-1]}).pdf")

    if RUN:
        for lag in lags:
            x.append(df['meantemp'].iloc[lag:lag + 365])
        results = {**{'w' : w_list}, **{f'lag = {lag}': [] for lag in lags}}
        for i ,lag in zip(range(1, len(x)), lags):
            for w in w_list:
                results[f'lag = {lag}'].append(TiOT_lib.TAOT(x[0].to_list(), x[i].to_list(), w = w)[0])
            print(f'Complete lag = {lag}')
        save_result(results, result_file)
        plot_graph(results, plot_file, 'w', r"$w$", r'TAOT($w$)')
    else:
        results = read_result(result_file)
        plot_graph(results, plot_file, 'w', r"$w$", r'TAOT($w$)')

def main():
    dist_w_exp(RUN=True)

main()