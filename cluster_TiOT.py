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
from sklearn.metrics.cluster import rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from itertools import combinations


eps_global = 0.01
w_global = 10
k_global = 20
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, eps = eps_global, freq = k_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global, costmatrix=TiOT_lib.costmatrix0)[0]

def euclidean(X1, X2):
    return np.sqrt(np.sum((np.array(X1) - np.array(X2))**2))

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

def compute_distance(args):
    i, j, Xi, Xj, metric = args
    return (i, j, metric(Xi, Xj))

def compute_distance_matrix(X, metric):
    n = len(X)
    total_pairs = (n * (n - 1)) // 2  # number of unique i < j pairs
    pairs_gen = [(i, j, X[i], X[j], metric) for i, j in combinations(range(n), 2)]  # generator
    #print(pairs_gen)
    D = np.zeros((n, n))
    # for a in Pool(5).imap(compute_pair, pairs_gen):
    #     print(a)
    print(pairs_gen[:0])
    with multiprocessing.Pool(32) as pool:
        for i, j, d in tqdm(pool.imap(compute_distance, pairs_gen), total=total_pairs):
            D[i, j] = D[j, i] = d

    return D

def cluster(dataset_name, data, metric_name , eps , w ):
    global w_global, eps_global
    w_global = w
    eps_global = eps
    if metric_name == "oriTAOT":
        metric = oriTAOT
    elif metric_name == "eTiOT":
        metric = eTiOT
    elif metric_name == 'euclidean':
        metric = euclidean
    elif metric_name == 'eTAOT':
        metric = eTAOT

    X = data[0]
    y = data[1]
    distance_matrix = compute_distance_matrix(X, metric)
    print('number of cluseter ', len(set(y)))
    agglo = AgglomerativeClustering(
    n_clusters=len(set(y)),
    metric='precomputed',  # For older versions < 1.2
    linkage='complete'        # Only 'average' or 'complete' are valid with precomputed distances
    )
    y_pred = agglo.fit_predict(distance_matrix)
    accuracy = rand_score(y, y_pred)
    error = 1 - accuracy
    # print(f"================> Predictions: {y_pred}\n\n")
    # print(f"================> True labels: {y}\n\n")
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)
    return error

def cluster_kMean(dataset_name, data, metric_name , eps , w ):
    X = data[0]
    y = data[1]
    print('number of cluseter ', len(set(y)))
    kmeans = KMeans(n_clusters=len(set(y)), random_state=0)
    y_pred = kmeans.fit_predict(X)
    accuracy = rand_score(y, y_pred)
    error = 1 - accuracy
    # print(f"================> Predictions: {y_pred}\n\n")
    # print(f"================> True labels: {y}\n\n")
    print(f"  ====>  Completed dataset: {dataset_name} via kMeans, Error:",error)
    return error    

def plot_results(results, plot_file):
    eps_list = results['eps']
    alg_names = [k for k in results.keys() if k != 'eps']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-', '-', '-', '-', '-']
    i = 0
    for name in alg_names: 
        if name != 'euclid' and name != 'kMeans':
            plt.plot(eps_list, results[name], label = name, linewidth=1.75, marker=markers[i], linestyle = linestyles[i])
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

def experiment_cluster(dataset_name, w_TAOT, RUN = True):
    eps_list = [0.01, 0.04, 0.07, 0.1]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"       
    plot_file = os.path.join("KMeans_data","plots", "Comparison on " + dataset_name + eps_name + ".pdf")
    result_file = os.path.join("KMeans_data", "saved_results","Results on " + dataset_name + eps_name + '.csv')
    if RUN :
        data = process_data(dataset_name= dataset_name)
        #cluster(dataset_name, data, metric_name='oriTAOT', eps =0.1, w = 2)
        w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
        alg_names = ["eTiOT"]  +  [f"eTAOT(w = {w})" for w in w_list] + ['euclid', 'kMeans']
        results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
        results['euclid']= cluster(dataset_name, data, metric_name='euclidean', eps = None, w = w_TAOT)
        results['kMeans'] = cluster_kMean(dataset_name, data, metric_name='euclidean', eps = None, w = w_TAOT)
        for eps in eps_list:
            #results['eTiOT'].append(cluster(dataset_name, data, metric_name='eTiOT', eps = eps, w = w_TAOT))
            results['eTiOT'].append(cluster(dataset_name, data, metric_name='eTiOT', eps = eps, w = w_TAOT))
            for w in w_list:
                results[f"eTAOT(w = {w})"].append(cluster(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w))
        save_result(results, result_file)
        plot_results(results, plot_file)
    else:
        results = read_result(result_file)
        plot_results(results, plot_file)
 
if __name__ == "__main__":
    # experiment_cluster("SonyAIBORobotSurface1", 2)
    # experiment_cluster("CBF", 1)
    # experiment_cluster("DistalPhalanxOutlineAgeGroup", 1)
    # experiment_cluster("ProximalPhalanxTW", 0.7)
    # experiment_cluster('ProximalPhalanxOutlineCorrect', 0.7)
    # experiment_cluster('ProximalPhalanxOutlineAgeGroup', 0.1)
    # experiment_cluster('MiddlePhalanxOutlineCorrect', 0.5)

    experiment_cluster('Adiac',0.1)
    experiment_cluster('SwedishLeaf',0.9)
    experiment_cluster('Chinatown', 1)
    
    # experiment_cluster("ECG200", 3)
    #experiment_cluster('SyntheticControl', 4)
    #experiment_cluster('ItalyPowerDemand', 7)
    #experiment_cluster('MoteStrain', 1)
    #experiment_cluster('ECGFiveDays', 5)


    # experiment_cluster('TwoLeadECG', 0.1)
    # experiment_cluster('MedicalImages', 4)
    # experiment_cluster('ArrowHead', 3)
    # experiment_cluster('ToeSegmentation2', 0.8)
    # experiment_cluster('ToeSegmentation1', 0.1)
    # experiment_cluster('Meat', 0.9)
    # experiment_cluster('ShapeletSim', 2)
    # experiment_cluster('DiatomSizeReduction', 0.2)
    # experiment_cluster('Ham', 0.7) unreasonable long running time
    # experiment_cluster('Wine', 9)
    # experiment_cluster('Car', 0.8)
    # experiment_cluster('Beef', 6)
    # experiment_cluster('Symbols', 0.8)
    # experiment_cluster('Strawberry', 0.2)
