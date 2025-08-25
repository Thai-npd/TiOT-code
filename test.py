import numpy as np
from typing import Callable, Iterable, Any, Dict, Tuple, List
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import TiOT_lib

# ---- module-level globals for worker processes (set via initializer) ----
_METRIC_FUNC = None
_X = None
_Y = None
_K = None
_RANDOM_STATE = None

eps_global = 0.01
w_global = 10
k_global = 20
def eTiOT(X1, X2):
    return TiOT_lib.eTiOT(X1,X2, eps=eps_global, freq=k_global)[0]

def eTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global)[0]

def oriTAOT(X1, X2):
    return TiOT_lib.eTAOT(X1,X2, w = w_global, eps = eps_global, costmatrix=TiOT_lib.costmatrix0)[0]

def _worker_init(metric_func, X, y, k, random_state):
    """Initializer for worker processes: set globals once per worker."""
    global _METRIC_FUNC, _X, _Y, _K, _RANDOM_STATE
    _METRIC_FUNC = metric_func
    _X = X
    _Y = y
    _K = k
    _RANDOM_STATE = random_state

def _evaluate_eps(eps) -> Tuple[Any, float]:
    """
    Worker: evaluate one eps via KFold CV (plain KFold since labels are balanced).
    Returns (eps, mean_accuracy).
    """
    global _METRIC_FUNC, _X, _Y, _K, _RANDOM_STATE, w_global, eps_global
    cv = KFold(n_splits=_K, shuffle=True, random_state=_RANDOM_STATE)
    scores: List[float] = []
    for train_idx, val_idx in cv.split(_X):
        X_tr, y_tr = _X[train_idx], _Y[train_idx]
        X_val, y_val = _X[val_idx], _Y[val_idx]

        correct = 0
        for xi, yi in zip(X_val, y_val):
            # brute-force distances (metric must be picklable and top-level)
            dists = [float(_METRIC_FUNC(xi, xt, eps)) for xt in X_tr]
            nn = int(np.argmin(dists))
            if y_tr[nn] == yi:
                correct += 1
        scores.append(correct / len(y_val))
    return (eps, float(np.mean(scores)))

def tune_and_train_1nn_parallel(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray, Any], float],
    eps_candidates: Iterable[Any],
    k: int = 5,
    random_state: int | None = None,
    n_procs: int = 8,
    verbose: bool = True,
) -> Tuple[KNeighborsClassifier, Any, Dict[Any, float]]:
    """
    Parallel search over eps_candidates using multiprocessing.Pool and KFold CV.
    Returns: (fitted_1nn_classifier, best_eps, cv_results_dict)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    eps_list = list(eps_candidates)
    if len(eps_list) == 0:
        raise ValueError("eps_candidates must be non-empty")

    if verbose:
        print(f"Starting evaluation of {len(eps_list)} candidates on {n_procs} processes...")

    init_args = (metric_func, X, y, k, random_state)
    with Pool(processes=n_procs, initializer=_worker_init, initargs=init_args) as pool:
        results_iter = pool.imap(_evaluate_eps, eps_list)
        results = list(tqdm(results_iter, total=len(eps_list)))

    cv_results: Dict[Any, float] = {eps: acc for (eps, acc) in results}
    # pick best eps (highest mean accuracy); tie -> first in eps_list order
    best_eps = max(eps_list, key=lambda eps: cv_results[eps])

    # final classifier on full data using chosen best_eps
    def bound_metric(a, b, eps=best_eps):
        return metric_func(a, b, eps)

    clf = KNeighborsClassifier(n_neighbors=1, metric=bound_metric)
    clf.fit(X, y)

    if verbose:
        print("CV results (eps -> mean accuracy):")
        for eps in eps_list:
            print(f"  {eps}: {cv_results[eps]:.4f}")
        print("Best eps:", best_eps)

    return clf, best_eps, cv_results

def kNN(dataset_name, data, metric_name , eps , w):
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

    # Synthetic balanced dataset
    X, y = make_classification(n_samples=800, n_features=8, n_informative=6,
                               n_redundant=0, n_classes=4, weights=None, random_state=0)

    eps_candidates = [0.05 * i for i in range(1, 21)]
    clf, best_eps, cv_results = tune_and_train_1nn_parallel(
        X, y, metric, eps_candidates,
        k=5, random_state=0, n_procs=5, verbose=True
    )

    # Optional: parallel predictions per-sample using the same imap pattern
    # (keeps your syntax: with Pool(...); pool.imap(...))
    def _pred_worker_init(classifier):
        global _PRED_CLF
        _PRED_CLF = classifier

    def _predict_single(x):
        global _PRED_CLF
        return int(_PRED_CLF.predict([x])[0])

    X_test = X[:200]
    with Pool(processes=3, initializer=_pred_worker_init, initargs=(clf,)) as pool:
        y_pred_list = list(tqdm(pool.imap(_predict_single, X_test), total=len(X_test)))
    y_pred = np.array(y_pred_list)

    print("Example predictions (first 10):", y_pred[:10])
# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
