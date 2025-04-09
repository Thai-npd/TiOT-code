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


eps_global = 0.01
w_global = 10
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global)[0]

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
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    with multiprocessing.Pool(50) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)
    return error

def experiment_kNN(dataset_name, eps_TAOT, w_TAOT ):
    data = process_data(dataset_name= dataset_name)
    errors = []
    eps_list = [-1, -1.4, -1.8, -2]
    errors.append(kNN(dataset_name, data, metric_name='oriTAOT', eps = eps_TAOT, w = w_TAOT))
    csv_filename = "euclid_kNN.csv"

    # Append data to the CSV file
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([dataset_name] + errors)
    print("Complete saving results")

def experiment_kNNgraph(dataset_name, w_TAOT):
    data = process_data(dataset_name= dataset_name)
    w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
    eps_list = [0.01*i for i in range(1,4)]
    alg_names = ["TiOT"] +  [f"TAOT(w = {w})" for w in w_list]
    results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
    for eps in eps_list:
        results['TiOT'].append(kNN(dataset_name, data, metric_name='eTiOT', eps = eps, w = w_TAOT))
        for w in w_list:
            results[f"TAOT(w = {w})"].append(kNN(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w))

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
    experiment_kNNgraph("SonyAIBORobotSurface1", 2)
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


    # experiment_kNN("SyntheticControl", 1/195, 4)
    # experiment_kNN("SonyAIBORobotSurface2", 1/105,10)
    # experiment_kNN("SonyAIBORobotSurface1", 1/95, 2)
    # experiment_kNN("ProximalPhalanxTW", 1/15, 0.7)
    # experiment_kNN("ProximalPhalanxOutlineCorrect", 1/40, 0.7)
    # experiment_kNN("ProximalPhalanxOutlineAgeGroup", 1/15, 0.1)
    # experiment_kNN("MiddlePhalanxTW", 1/70, 0.4)
    # experiment_kNN("MiddlePhalanxOutlineCorrect", 1/20, 0.5)
    # experiment_kNN("MiddlePhalanxOutlineAgeGroup", 1/60, 0.2)
    # experiment_kNN("GunPoint", 1/50, 0.3)
    # experiment_kNN("DistalPhalanxTW", 1/5, 0.5)
    # experiment_kNN("DistalPhalanxOutlineCorrect", 1/45, 0.4)
    # experiment_kNN("DistalPhalanxOutlineAgeGroup", 1/5, 1)
    # experiment_kNN("CBF", 1/145, 1)
    # experiment_kNN("Chinatown", 1/100, 1)
    # experiment_kNN("FaceFour", 1/40, 5 )