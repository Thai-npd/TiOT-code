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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

eps_global = 0.01
w_global = 10
k_global = 20
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global, freq=k_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global, costmatrix=TiOT_lib.costmatrix0)[0]

def process_data(dataset_name ):
    train_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TRAIN.txt" )
    test_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TEST.txt")

    with open(train_file, "r") as file:
        data = np.array([line.strip().split() for line in file], dtype=float)

    # # Convert to numerical values if needed
    # data = [[float(value) for value in row] for row in data]

    Y_train = data[:, 0]      # first column
    X_train = data[:, 1:]     # all columns except the first


    with open(test_file, "r") as file:
        data_test = np.array([line.strip().split() for line in file], dtype=float)


    Y_test = data_test[:, 0]
    X_test = data_test[:, 1:] 

    return [X_train, Y_train, X_test, Y_test]

def my_cross_val_score(model, X, y, cv=3):
    """
    Replicates sklearn's cross_val_score for KNN without parallelization.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.
    cv : int
        Number of folds for cross-validation.
    n_neighbors : int
        Number of neighbors for KNN.
    
    Returns
    -------
    scores : list of float
        Accuracy scores for each fold.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    scores = []

    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit KNN
        model.fit(X_train, y_train)

        # Evaluate
        with multiprocessing.Pool(64) as pool:
            y_pred = list(tqdm(pool.imap(model.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    error = 1 - np.array(scores).mean()
    return error

def kNN(dataset_name, data, metric_name , eps_list , w ):
    global w_global, eps_global
    w_global = w
    if metric_name == "oriTAOT":
        metric = oriTAOT
    elif metric_name == "eTiOT":
        metric = eTiOT
    elif metric_name == 'euclidean':
        metric = 'euclidean'
    elif metric_name == 'eTAOT':
        metric = eTAOT

    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    eps_best = 0
    error_best = np.inf
    errors = []
    for eps in eps_list:
        eps_global = eps
        print(f"Start cross validation with metric = {metric_name}, eps = {eps}")
        knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
        error = my_cross_val_score(knn, X_train, Y_train, cv = 3)
        errors.append(error)
        if error_best >= error:
            eps_best = eps
            error_best = error
    eps_global = eps_best
    print(f"After cross validation eps_best = {eps_best} with average error {error_best}")
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    with multiprocessing.Pool(64) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    final_error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, eps = {eps_best}, Error:",final_error)
    errors.append(final_error)
    return errors

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
    eps_list = [0.005*i for i in range(1,21)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"       
    plot_file = os.path.join("kfold_kNN_data","plots", "Comparison on " + dataset_name + eps_name  + '_new_' + ".pdf")
    result_file = os.path.join("kfold_kNN_data", "saved_results","Results on " + dataset_name + eps_name +  '_new_' + '.csv')
    if RUN :
        data = process_data(dataset_name = dataset_name)
        results = {**{'eps': eps_list}}
        results['eTiOT'] = kNN(dataset_name, data, metric_name='eTiOT', eps_list= eps_list, w = None)
        results['eTAOT'] = kNN(dataset_name, data, metric_name='oriTAOT', eps_list= eps_list, w = w_TAOT)
        results['eps'].append('Final error')

        save_result(results, result_file)
        plot_results(results, plot_file)
    else:
        results = read_result(result_file)
        plot_results(results, plot_file)
 
if __name__ == "__main__":
    # ===> Tier 1 

    experiment_kNN("SonyAIBORobotSurface1", 2)
    experiment_kNN("ProximalPhalanxTW", 0.7)
    experiment_kNN('ProximalPhalanxOutlineCorrect', 0.7)
    experiment_kNN("DistalPhalanxOutlineAgeGroup", 1)
    experiment_kNN('MiddlePhalanxOutlineCorrect', 0.5)
    # experiment_kNN('DistalPhalanxOutlineCorrect', 0.4)
    # experiment_kNN('MiddlePhalanxOutlineAgeGroup', 0.2)
    # experiment_kNN('MiddlePhalanxTW', 0.4)
    # experiment_kNN("CBF", 1)
    # experiment_kNN('SwedishLeaf',0.9) 
    # experiment_kNN('Adiac',0.1) 

    # ==> New data
    # experiment_kNN('DistalPhalanxTW', 0.5 )
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
    

    # experiment_kNN('ProximalPhalanxOutlineAgeGroup', 0.1)
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
