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

# dataset_name = "SyntheticControl"
# eps = 0.01
# w = 0.5

eps_global = 0.01
w_global = 10
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, lamda=1/eps_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, lamda = 1/eps_global, costmatrix=TiOT_lib.costmatrix1)[0]

# metric = TiOT

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
    #pool = multiprocessing.Pool(processes = 100)
    #y_pred = pool.map(knn.predict, [[x_test] for x_test in X_test])
    with multiprocessing.Pool(50) as pool:
        y_pred = list(tqdm(pool.imap(knn.predict, [[x_test] for x_test in X_test]), total=len(X_test)))
    pool.close()
    accuracy = accuracy_score(Y_test, y_pred)
    error = 1 - accuracy
    print(f"  ====>  Completed dataset: {dataset_name}, Metric : {metric_name}, Error:",error)

    # # CSV file path
    # csv_filename = "kNN_report.csv"

    # # Append data to the CSV file
    # with open(csv_filename, mode="a", newline="") as file:
    #     writer = csv.writer(file)
        
    #     # If the file is empty, write the header
    #     if file.tell() == 0:
    #         writer.writerow(["name_dataset", "name_metric", "error"])
        
    #     writer.writerow([dataset_name, metric.__name__, error])
    # print("Complete saving results")
    return error

def experiment_kNN(dataset_name, eps_TAOT, w_TAOT ):
    data = process_data(dataset_name= dataset_name)
    errors = []
    #eps_list = [-1.2, -1.4,-1.6,-1.8,-2,-2.2]
    eps_list = [-1, -1.4, -1.8, -2]
    # errors.append(kNN(dataset_name, data, metric_name=4, eps = eps_TAOT, w = w_TAOT))
    errors.append(kNN(dataset_name, data, metric_name='oriTAOT', eps = eps_TAOT, w = w_TAOT))
    # for eps in [10**i for i in eps_list ]:
    #     errors.append(kNN(dataset_name, data, metric_name= 2, eps = eps, w=None))

    # CSV file path
    csv_filename = "euclid_kNN.csv"

    # Append data to the CSV file
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header
        # if file.tell() == 0:
        #     writer.writerow(["name_dataset", "oriTAOT"]  +  ["TiOT(eps = 10^" + str(eps) + ")" for eps in eps_list])
        
        writer.writerow([dataset_name] + errors)
    print("Complete saving results")

if __name__ == "__main__":
    experiment_kNN("SyntheticControl", 1/195, 4)
    experiment_kNN("SonyAIBORobotSurface2", 1/105,10)
    experiment_kNN("SonyAIBORobotSurface1", 1/95, 2)
    experiment_kNN("ProximalPhalanxTW", 1/15, 0.7)
    experiment_kNN("ProximalPhalanxOutlineCorrect", 1/40, 0.7)
    experiment_kNN("ProximalPhalanxOutlineAgeGroup", 1/15, 0.1)
    experiment_kNN("MiddlePhalanxTW", 1/70, 0.4)
    experiment_kNN("MiddlePhalanxOutlineCorrect", 1/20, 0.5)
    experiment_kNN("MiddlePhalanxOutlineAgeGroup", 1/60, 0.2)
    experiment_kNN("GunPoint", 1/50, 0.3)
    experiment_kNN("DistalPhalanxTW", 1/5, 0.5)
    experiment_kNN("DistalPhalanxOutlineCorrect", 1/45, 0.4)
    experiment_kNN("DistalPhalanxOutlineAgeGroup", 1/5, 1)
    experiment_kNN("CBF", 1/145, 1)
    experiment_kNN("Chinatown", 1/100, 1)
    experiment_kNN("FaceFour", 1/40, 5 )