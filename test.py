import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from sklearn.datasets import load_iris

def compute_pair(args):
    i, j, Xi, Xj = args
    return (i, j, euclidean(Xi, Xj))

def parallel_distance_matrix_with_progress(X):
    n = len(X)
    total_pairs = (n * (n - 1)) // 2  # number of unique i < j pairs
    pairs_gen = [(i, j, X[i], X[j]) for i, j in combinations(range(n), 2)]  # generator
    #print(pairs_gen)
    D = np.zeros((n, n))
    # for a in Pool(5).imap(compute_pair, pairs_gen):
    #     print(a)
    with Pool(5) as pool:
        for i, j, d in tqdm(pool.imap(compute_pair, pairs_gen), total=total_pairs):
            D[i, j] = D[j, i] = d

    return D

def main():
    X = load_iris().data
    D = parallel_distance_matrix_with_progress(X)

if __name__ == '__main__':
    main()