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
import time


eps_global = 0.01
w_global = 10
k_global = 20
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global)[0]

def fast_eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global, w_update_freq=k_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, eps = eps_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global, costmatrix=TiOT_lib.costmatrix0)[0]

def process_data(dataset_name ):
    train_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TRAIN.txt" )
    test_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TEST.txt")

    with open(train_file, "r") as file:
        data = [line.strip().split() for line in file]

    # Convert to numerical values if needed
    data = [[float(value) for value in row] for row in data]

    Y_train = [row[0] for row in data]
    X_train = [row[1:] for row in data]

    with open(test_file, "r") as file:
        data_test = [line.strip().split() for line in file]

    # Convert to numerical values if needed
    data_test = [[float(value) for value in row] for row in data_test]

    Y_test = [row[0] for row in data_test]
    X_test = [row[1:] for row in data_test]
    return [X_train, Y_train, X_test, Y_test]

def kNN(dataset_name, data, metric_name , eps , w ):
    global w_global, eps_global
    w_global = w
    eps_global = eps
    if metric_name == "oriTAOT":
        metric = oriTAOT
    elif metric_name == "eTiOT":
        metric = eTiOT
    elif metric_name == 'euclidean':
        metric = 'euclidean'
    elif metric_name == 'eTAOT':
        metric = eTAOT
    elif metric_name == f'eTiOT(k = {k_global})':
        metric = fast_eTiOT
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    start_time = time.perf_counter()
    with multiprocessing.Pool(5) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    end_time = time.perf_counter()
    accuracy = accuracy_score(Y_test, y_pred)
    error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)
    return [error, end_time - start_time]

def plot_results(results, plot_file):
    eps_list = results['eps']
    alg_names = [k for k in results.keys() if k != 'eps']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    i = 0
    for name in alg_names:
        #plt.scatter(eps_list, results[name], marker=markers[i])
        print(np.array(results[name]))
        plt.plot(eps_list, np.array(results[name])[:,0], label = name, linewidth=1.75, marker=markers[i])
        i+=1
    plt.xlabel(r"$\varepsilon$", fontsize = 14)
    plt.ylabel("Error", fontsize = 14)
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

def experiment_kNNgraph(dataset_name, w_TAOT, RUN = True):
    eps_list = [0.01*i for i in range(1,11)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"       
    plot_file = os.path.join("kNN_data","plots", "Comparison on " + dataset_name + eps_name + ".pdf")
    result_file = os.path.join("kNN_data", "saved_results","Results on " + dataset_name + eps_name + '.csv')
    time_record_file = os.path.join("kNN_data", "saved_results","Time record of " + dataset_name + eps_name + '.csv')
    if RUN :
        data = process_data(dataset_name= dataset_name)
        w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
        alg_names = ["eTiOT", f"eTiOT(k = {k_global})"]  +  [f"eTAOT(w = {w})" for w in w_list]
        results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
        time_record = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
        for eps in eps_list:
            error, elapsed_time = kNN(dataset_name, data, metric_name='eTiOT', eps = eps, w = w_TAOT)
            results['eTiOT'].append(error)
            time_record['eTiOT'].append(elapsed_time)
            error, elapsed_time = kNN(dataset_name, data, metric_name=f'eTiOT(k = {k_global})', eps = eps, w = w_TAOT)
            results[f'eTiOT(k = {k_global})'].append(error)
            time_record[f'eTiOT(k = {k_global})'].append(elapsed_time)
            for w in w_list:
                error, elapsed_time = kNN(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w)
                results[f"eTAOT(w = {w})"].append(error)
                time_record[f"eTAOT(w = {w})"].append(elapsed_time)

        save_result(results, result_file)
        plot_results(results, plot_file)
    else:
        results = read_result(result_file)
        plot_results(results, plot_file)
 
if __name__ == "__main__":
    # experiment_kNNgraph("CBF", 1)
    # experiment_kNNgraph("DistalPhalanxOutlineAgeGroup", 1)
    experiment_kNNgraph("SonyAIBORobotSurface1", 2, RUN=False)
    # experiment_kNNgraph("ProximalPhalanxTW", 0.7)
    # experiment_kNNgraph('ProximalPhalanxOutlineCorrect', 0.7)
    # experiment_kNNgraph('ProximalPhalanxOutlineAgeGroup', 0.1)
    # experiment_kNNgraph('MiddlePhalanxOutlineCorrect', 0.5)

    #experiment_kNNgraph('Adiac',0.1)
    #experiment_kNNgraph("ECG200", 3)
    #experiment_kNNgraph('SwedishLeaf',0.9)
    #experiment_kNNgraph('SyntheticControl', 4)
    #experiment_kNNgraph('Chinatown', 1)
    #experiment_kNNgraph('ItalyPowerDemand', 7)
    #experiment_kNNgraph('MoteStrain', 1)
    #experiment_kNNgraph('ECGFiveDays', 5)


    # experiment_kNNgraph('TwoLeadECG', 0.1)
    # experiment_kNNgraph('MedicalImages', 4)
    # experiment_kNNgraph('ArrowHead', 3)
    # experiment_kNNgraph('ToeSegmentation2', 0.8)
    # experiment_kNNgraph('ToeSegmentation1', 0.1)
    # experiment_kNNgraph('Meat', 0.9)
    # experiment_kNNgraph('ShapeletSim', 2)
    # experiment_kNNgraph('DiatomSizeReduction', 0.2)
    # experiment_kNNgraph('Ham', 0.7) unreasonable long running time
    # experiment_kNNgraph('Wine', 9)
    # experiment_kNNgraph('Car', 0.8)
    # experiment_kNNgraph('Beef', 6)
    # experiment_kNNgraph('Symbols', 0.8)
    # experiment_kNNgraph('Strawberry', 0.2)
