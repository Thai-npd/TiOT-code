# Keep each worker's numeric libraries single-threaded so the process-level pool
# below does not oversubscribe the cores (one BLAS thread per worker process).
# Must run before numpy / TiOT_lib import their backends.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import TiOT_lib
import multiprocessing
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy import stats
from tqdm import tqdm

# Number of worker processes. The cross-validation phase has (metrics x seeds x eps)
# independent heavy tasks, so this scales well up to that count.
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# ---------------------------------------------------------------------------
# Metrics.
#
# eps and w are passed EXPLICITLY as arguments and bound with functools.partial.
# partial is picklable and carries its bound values as data, so the correct eps/w
# reach the worker processes under any start method (spawn on Windows/macOS as
# well as fork on Linux). This replaces the previous module-global approach, which
# silently used the import-time eps/w inside spawned workers.
# ---------------------------------------------------------------------------
def eTiOT_metric(X1, X2, eps):
    return TiOT_lib.eTiOT(X1, X2, eps=eps)[0]

def eTAOT_metric(X1, X2, eps, w):
    return TiOT_lib.eTAOT(X1, X2, w=w, eps=eps)[0]

def oriTAOT_metric(X1, X2, eps, w):
    return TiOT_lib.eTAOT(X1, X2, w=w, eps=eps, costmatrix=TiOT_lib.costmatrix0)[0]

def build_metric(metric_name, eps, w):
    if metric_name == 'eTiOT':
        return partial(eTiOT_metric, eps=eps)
    elif metric_name == 'oriTAOT':
        return partial(oriTAOT_metric, eps=eps, w=w)
    elif metric_name == 'eTAOT':
        return partial(eTAOT_metric, eps=eps, w=w)
    elif metric_name == 'euclidean':
        return 'euclidean'
    raise ValueError(f"Unknown metric_name: {metric_name}")


def process_data(dataset_name):
    train_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TRAIN.txt")
    test_file = os.path.join("time_series_kNN", dataset_name, dataset_name + "_TEST.txt")

    with open(train_file, "r") as file:
        data = np.array([line.strip().split() for line in file], dtype=float)

    Y_train = data[:, 0]
    X_train = data[:, 1:]

    with open(test_file, "r") as file:
        data_test = np.array([line.strip().split() for line in file], dtype=float)

    Y_test = data_test[:, 0]
    X_test = data_test[:, 1:]

    return [X_train, Y_train, X_test, Y_test]


def confidence_interval(values, confidence=0.95):
    """95% confidence interval of the mean using the Student-t distribution."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    if n < 2:
        return mean, 0.0, mean, mean
    sem = arr.std(ddof=1) / np.sqrt(n)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width, mean - half_width, mean + half_width


def cross_val_error(metric, X, y, seed, cv=3):
    """Mean 1-NN cross-validation error for a fully-specified metric (eps/w already bound)."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(X):
        knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
        knn.fit(X[train_idx], y[train_idx])
        y_pred = knn.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], y_pred))
    return 1 - float(np.mean(scores))


# ---------------------------------------------------------------------------
# Worker side. The dataset is loaded once per worker via the pool initializer
# (sent once at pool creation) rather than pickled with every task.
# ---------------------------------------------------------------------------
_DATA = None

def _init_worker(data):
    global _DATA
    _DATA = data

def _cv_task(task):
    """One full k-fold cross-validation for a (metric, eps, w, seed) combination."""
    metric_name, eps, w, seed = task
    X_train, Y_train = _DATA[0], _DATA[1]
    metric = build_metric(metric_name, eps, w)
    return cross_val_error(metric, X_train, Y_train, seed=seed)

def _test_task(task):
    """1-NN prediction for a single test sample at a fixed (metric, eps, w)."""
    metric_name, eps, w, i = task
    X_train, Y_train, X_test = _DATA[0], _DATA[1], _DATA[2]
    metric = build_metric(metric_name, eps, w)
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    return knn.predict([X_test[i]])[0]


def select_eps_candidates(errors, eps_list):
    """Return (smallest_eps, largest_eps) among the eps that achieve the minimum
    validation error. When the minimiser is unique (no tie) the two are equal, so
    the test phase evaluates a single eps for that (metric, seed)."""
    best_err = min(errors)
    tied = [eps for eps, err in zip(eps_list, errors) if err == best_err]
    return min(tied), max(tied)


def save_report(dataset_name, eps_list, seeds, metric_reports, report_file):
    lines = []
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"eps grid: {eps_list}")
    lines.append(f"Seeds: {list(seeds)}")
    lines.append("")

    for metric_label, rep in metric_reports.items():
        lines.append(f"===== Metric: {metric_label} =====")

        val_df = pd.DataFrame(
            {f"seed{s}": rep['val_errors'][s] for s in seeds},
            index=[f"eps={e}" for e in eps_list],
        )
        lines.append("Validation errors (rows = eps, cols = seed):")
        lines.append(val_df.to_string())
        lines.append("")

        # Two tie-break rules, each with its own selected eps and 95% CI, so the
        # better-performing rule can be chosen later from these numbers.
        for rule_label, eps_key, final_key in [
            ("tie-break = smallest eps", 'eps_small', 'final_small'),
            ("tie-break = largest eps",  'eps_large', 'final_large'),
        ]:
            eps_str = ", ".join(f"seed{s}={rep[eps_key][s]}" for s in seeds)
            finals = [rep[final_key][s] for s in seeds]
            mean, half_width, lo, hi = confidence_interval(finals)
            lines.append(f"--- {rule_label} ---")
            lines.append(f"  Selected eps per seed  : {eps_str}")
            lines.append(f"  Final test error / seed: {finals}")
            lines.append(f"  Mean final test error  : {mean:.6f}")
            lines.append(f"  95% confidence interval: [{lo:.6f}, {hi:.6f}]  (mean +/- {half_width:.6f})")
            lines.append("")

    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {report_file}")


def experiment_kNN(dataset_name, w_TAOT, seeds=range(1, 6)):
    seeds = list(seeds)
    eps_list = [round(0.01 * i, 2) for i in range(1, 11)]
    eps_name = f" ({eps_list[0]} to {eps_list[-1]})"
    result_file = os.path.join('Experimental_outputs', "kfold_kNN_data", "saved_results",
                               "Results on " + dataset_name + eps_name + '.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    data = process_data(dataset_name=dataset_name)
    Y_test = data[3]
    n_test = len(Y_test)

    metrics = [
        {'key': 'eTiOT', 'label': 'eTiOT',                'metric_name': 'eTiOT',   'w': None},
        {'key': 'eTAOT', 'label': f'eTAOT (w={w_TAOT})',  'metric_name': 'oriTAOT', 'w': w_TAOT},
    ]

    # Build the whole cross-validation work-list up front: one task per
    # (metric, seed, eps). These are independent and heavy -> ideal for one pool.
    cv_tasks = [(m['metric_name'], eps, m['w'], seed)
                for m in metrics for seed in seeds for eps in eps_list]

    with multiprocessing.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        # ---- Phase 1: cross-validation ----
        cv_errors = list(tqdm(pool.imap(_cv_task, cv_tasks),
                              total=len(cv_tasks), desc="cross-val"))

        # Reshape flat results back into val_errors[key][seed] = [error per eps].
        val_errors = {m['key']: {s: [] for s in seeds} for m in metrics}
        idx = 0
        for m in metrics:
            for s in seeds:
                for _ in eps_list:
                    val_errors[m['key']][s].append(cv_errors[idx])
                    idx += 1

        # Two tie-break candidates per (metric, seed): the smallest and the largest eps
        # achieving the minimum validation error. Both are evaluated on the test set so
        # the winning rule can be decided later; they coincide when there is no tie.
        eps_small, eps_large = {}, {}
        for m in metrics:
            for s in seeds:
                lo, hi = select_eps_candidates(val_errors[m['key']][s], eps_list)
                eps_small[(m['key'], s)] = lo
                eps_large[(m['key'], s)] = hi

        # ---- Phase 2: final test error at each selected eps ----
        # Unique (metric, eps) combinations only -> the test set is deterministic in eps,
        # so seeds/rules landing on the same eps share one evaluation. Flatten to one task
        # per (combination, test sample) so the whole test phase saturates the pool too.
        combos = {}  # (key, eps) -> (metric_name, w)
        for m in metrics:
            for s in seeds:
                for eps in (eps_small[(m['key'], s)], eps_large[(m['key'], s)]):
                    combos[(m['key'], eps)] = (m['metric_name'], m['w'])
        combo_list = list(combos.items())

        test_tasks = [(metric_name, eps, w, i)
                      for (key, eps), (metric_name, w) in combo_list
                      for i in range(n_test)]
        test_preds = list(tqdm(pool.imap(_test_task, test_tasks),
                               total=len(test_tasks), desc="test"))

    # Aggregate per-combo test error (walk test_preds in the same order it was built).
    test_error = {}
    pos = 0
    for (key, eps), (metric_name, w) in combo_list:
        preds = test_preds[pos:pos + n_test]
        pos += n_test
        test_error[(key, eps)] = 1 - accuracy_score(Y_test, preds)

    # Assemble the per-metric report (both tie-break rules).
    metric_reports = {}
    for m in metrics:
        key = m['key']
        rep = {'val_errors': {}, 'eps_small': {}, 'eps_large': {},
               'final_small': {}, 'final_large': {}}
        for s in seeds:
            rep['val_errors'][s] = val_errors[key][s]
            rep['eps_small'][s] = eps_small[(key, s)]
            rep['eps_large'][s] = eps_large[(key, s)]
            rep['final_small'][s] = test_error[(key, eps_small[(key, s)])]
            rep['final_large'][s] = test_error[(key, eps_large[(key, s)])]
        metric_reports[m['label']] = rep

    save_report(dataset_name, eps_list, seeds, metric_reports, result_file)


if __name__ == "__main__":
    # experiment_kNN('Adiac', 0.1)
    # experiment_kNN('ArrowHead', 3)
    # experiment_kNN("CBF", 1)
    # experiment_kNN('BirdChicken', 0.1)
    # experiment_kNN("DistalPhalanxOutlineAgeGroup", 1)
    # experiment_kNN('DistalPhalanxOutlineCorrect', 0.4)
    # experiment_kNN('DistalPhalanxTW', 0.5 )
    # experiment_kNN('Ham', 0.7)
    # experiment_kNN('MiddlePhalanxOutlineAgeGroup', 0.2)
    # experiment_kNN('MiddlePhalanxOutlineCorrect', 0.5)
    # experiment_kNN('MiddlePhalanxTW', 0.4)
    # experiment_kNN('ProximalPhalanxOutlineCorrect', 0.7)
    # experiment_kNN("ProximalPhalanxTW", 0.7)
    experiment_kNN("SonyAIBORobotSurface1", 2)
    # experiment_kNN('SwedishLeaf',0.9)
