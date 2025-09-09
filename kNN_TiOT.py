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
freq_global = 5
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global, freq=freq_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global)[0]

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

def get_w_opt(X_train, Y_train):
    np.random.seed(0)
    # unique_labels = np.unique(Y_train)
    # label_1, label_2 = unique_labels[0], unique_labels[1]

    # dist_min = np.inf
    # w_opt = 0
    # label_1_index = np.where(Y_train == label_1)[0]
    # label_2_index = np.where(Y_train == label_2)[0]
    # n_pairs = min(len(label_1_index), len(label_2_index))
    # for _ in range(n_pairs):
    #     idx1 = np.random.choice(label_1_index)
    #     idx2 = np.random.choice(label_2_index)
    #     distance, plan, w = TiOT_lib.eTiOT(X_train[idx1], X_train[idx2], eps=eps, freq=k_global)
    #     if dist_min > distance:
    #         dist_min = distance
    #         w_opt = w
    # print(f"===> w_opt = {w_opt}")

    dist_min = np.inf
    w_opt = 0
    all_pairs = [(i, j) for i in range(len(Y_train)) for j in range(len(Y_train)) if i < j and Y_train[i] != Y_train[j]]
    selected_pairs = [all_pairs[idx] for idx in np.random.choice(len(all_pairs), size=len(Y_train), replace=False)]
    for idx1, idx2 in selected_pairs:
        distance, plan, w = TiOT_lib.eTiOT(X_train[idx1], X_train[idx2], eps=0.01, freq=5)
        if dist_min > distance:
            dist_min = distance
            w_opt = w
    print(f"===> w_opt = {w_opt}")
    return w_opt

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
    with multiprocessing.Pool(32) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)
    return error

def plot_results(results, plot_file):
    eps_list = results['eps']
    alg_names = [k for k in results.keys() if k != 'eps']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 5))
    markers = ['o', '^', 'D',  'v', 'P', 'X']
    linestyles = ['-', '-', "-", '-', '-']
    i = 0
    for name in alg_names:
        plt.plot(eps_list, np.array(results[name]), label = name, linewidth=1.75, marker=markers[i], linestyle = linestyles[i], markersize = 7)
        i+=1
    plt.xlabel(r"$\varepsilon$", fontsize = 16)
    plt.ylabel("Error", fontsize = 16)
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

def experiment_kNN(dataset_name, w_TAOT, RUN = True):
    eps_list = [0.01*i for i in range(1,11)]
    #eps_list = [0.005*i for i in range(1,21)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"       
    plot_file = os.path.join("kNN_data","plots", "Comparison on " + dataset_name + eps_name + f'_freq{freq_global}_'+  ".pdf")
    result_file = os.path.join("kNN_data", "saved_results","Results on " + dataset_name + eps_name + f'_freq{freq_global}_'  + '.csv')
    if RUN :
        data = process_data(dataset_name = dataset_name)
        w_list = [ round(w_TAOT/5, 3), w_TAOT,w_TAOT*5]
        w_list_name = [r'\omega_{\text{grid}} \;/\; 5', r'\omega_{\text{grid}}', r'\omega_{\text{grid}} \times 5']
        alg_names = ["eTiOT"]   +  [fr"eTAOT$(\omega = {w})$" for w in w_list_name]
        results = {**{'eps': eps_list}, **{name: [] for name in alg_names}}
        for eps in eps_list:
            results['eTiOT'].append(kNN(dataset_name, data, metric_name='eTiOT', eps = eps, w = None))
            for i in range(len(w_list)):
                results[fr"eTAOT$(\omega = {w_list_name[i]})$"].append(kNN(dataset_name, data, metric_name='oriTAOT', eps = eps, w = w_list[i]))

        save_result(results, result_file)
        plot_results(results, plot_file)
    else:
        results = read_result(result_file)
        plot_results(results, plot_file)
 
if __name__ == "__main__":
    # ===> Tier 1 

    # experiment_kNN("DistalPhalanxOutlineAgeGroup", 1, RUN=False)
    # experiment_kNN('DistalPhalanxOutlineCorrect', 0.4, RUN = False)
    # experiment_kNN('MiddlePhalanxOutlineAgeGroup', 0.2, RUN = False)
    # experiment_kNN('MiddlePhalanxOutlineCorrect', 0.5, RUN = False)
    # experiment_kNN('MiddlePhalanxTW', 0.4, RUN = False)
    # experiment_kNN('ProximalPhalanxOutlineCorrect', 0.7, RUN = False)
    experiment_kNN("ProximalPhalanxTW", 0.7, RUN = False)
    # experiment_kNN("SonyAIBORobotSurface1", 2)
    # experiment_kNN("CBF", 1)
    # experiment_kNN('SwedishLeaf',0.9) 
    
    # experiment_kNN('Adiac',0.1) 
    # ==> New data
    # experiment_kNN('DistalPhalanxTW', 0.5 )
    # experiment_kNN('ProximalPhalanxOutlineAgeGroup', 0.1)
    # experiment_kNN("SonyAIBORobotSurface2", 10)
    # experiment_kNN('Coffee', 2 )
    # experiment_kNN('Plane', 0.5)
    # experiment_kNN('BeetleFly', 0.3)
    # experiment_kNN('Herring', 0.2)
    # experiment_kNN('BirdChicken', 0.1)
    # experiment_kNN('Earthquakes', 7)
    # experiment_kNN('Lightning7', 0.9)



    # experiment_kNN('Car', 0.8)
    # experiment_kNN('MoteStrain', 1)
    # experiment_kNN('Trace', 0.3)
    

    # experiment_kNN("ECG200", 3)
    # experiment_kNN('ECGFiveDays', 5)
    # experiment_kNN('TwoLeadECG', 0.1)
    # experiment_kNN('SyntheticControl', 4)
    # experiment_kNN('Chinatown', 1)
    # experiment_kNN('ItalyPowerDemand', 7)
    # experiment_kNN('ToeSegmentation2', 0.8)
    # experiment_kNN('DistalPhalanxTW', 0.5)


    # experiment_kNN('MedicalImages', 4)
    # experiment_kNN('ArrowHead', 3)
    # experiment_kNN('ToeSegmentation1', 0.1)
    # experiment_kNN('Meat', 0.9)
    # experiment_kNN('ShapeletSim', 2)
    # experiment_kNN('DiatomSizeReduction', 0.2)
    # experiment_kNN('Ham', 0.7) unreasonable long running time
    # experiment_kNN('Wine', 9)
    # experiment_kNN('Beef', 6)
    # experiment_kNN('Symbols', 0.8)
    # experiment_kNN('Strawberry', 0.2)
