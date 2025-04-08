import numpy as np
import ot
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
import os
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import csv
import kNN_TiOT
from matplotlib.ticker import LogFormatterMathtext

def experiment_kNNgraph(dataset_name, w_TAOT):
    data = kNN_TiOT.process_data(dataset_name= dataset_name)
    #w_list = [0, w_TAOT/4, w_TAOT/2, w_TAOT,w_TAOT*2, w_TAOT*4, 7, 10]
    w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
    eps_list = [0.01*i for i in range(1,11)]
    alg_names = ["TiOT"] +  [f"TAOT(w = {w})" for w in w_list]
    results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
    for eps in eps_list:
        results['TiOT'].append(kNN_TiOT.kNN(dataset_name, data, metric_name='eTiOT', eps = eps, w = w_TAOT))
        for w in w_list:
            results[f"TAOT(w = {w})"].append(kNN_TiOT.kNN(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w))

    eps_name = str(eps_list[0]) + "-->" + str(eps_list[-1])         # str(exps[0]) + "-->" + str(exps[-1]) 
    plot_file = os.path.join("plots", "Comparison on " + dataset_name + " with " + eps_name + ".pdf")
    result_file = os.path.join("saved_results","Results on " + dataset_name + eps_name + '.csv')

    plt.figure()
    for name in alg_names:
        plt.scatter(eps_list, results[name])
        plt.plot(eps_list, results[name], label = name)
    plt.legend()
    plt.show()
    plt.savefig(plot_file)

    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())  # header
        rows = zip(*results.values())    # transpose
        writer.writerows(rows)  

if __name__ == "__main__":
    #experiment_kNNgraph("CBF", 1)
    #experiment_kNNgraph("DistalPhalanxOutlineAgeGroup", 1)
    #experiment_kNNgraph("SonyAIBORobotSurface1", 2)
    #experiment_kNNgraph("ProximalPhalanxTW", 0.7)
    #experiment_kNNgraph("ECG200", 3)
    #experiment_kNNgraph('SwedishLeaf',0.9)
    #experiment_kNNgraph('SyntheticControl', 4)
    #experiment_kNNgraph('Chinatown', 1)
    #experiment_kNNgraph('ItalyPowerDemand', 7)
    #experiment_kNNgraph('MoteStrain', 1)
    #experiment_kNNgraph('ECGFiveDays', 5)

    #experiment_kNNgraph('ProximalPhalanxOutlineCorrect', 0.7)
    #experiment_kNNgraph('ProximalPhalanxOutlineAgeGroup', 0.1)
    #experiment_kNNgraph('MiddlePhalanxOutlineCorrect', 0.5)
    # experiment_kNNgraph('TwoLeadECG', 0.1)
    # experiment_kNNgraph('MedicalImages', 4)
    experiment_kNNgraph('ArrowHead', 3)
    # experiment_kNNgraph('ToeSegmentation2', 0.8)
    # experiment_kNNgraph('ToeSegmentation1', 0.1)
    # experiment_kNNgraph('Meat', 0.9)
    # experiment_kNNgraph('ShapeletSim', 2)
    # experiment_kNNgraph('DiatomSizeReduction', 0.2)
    # experiment_kNNgraph('Ham', 0.7) unreasonable long running time
    # experiment_kNNgraph('Wine', 9)
    # experiment_kNNgraph('Car', 0.8)
    # experiment_kNNgraph('Beef', 6)
    experiment_kNNgraph('Symbols', 0.8)
    # experiment_kNNgraph('Strawberry', 0.2)
