This is the instruction file for the code of paper "COMPOSITE OPTIMIZATION MODELS VIA PROXIMAL GRADIENT METHOD WITH INCREASING ADAPTIVE STEPSIZES" by Pham Thi Hoai and Nguyen Pham Duy Thai

- There're five python files named by corresponding tested instances. 
- The "run_experiment.py" file is designed to run many problems, sizes, random datasets all at once.
- In the main function call the function correspond to the problem that you want to run for example : "experiment_lasso()", sizes and random seeds need to be specified inside each experiment function.

- Each *.py file is started with the control section, which need to be correctly set up:
    LOAD_DATA: if True, the code would find the dataset and use that data to run the algorithms, if False, then data would be created.
    SAVE_DATA: if True the data would be saved, noticed that sometimes data could be large and cost a significant amount of memory.
    LOAD_RESULTS: if True the results of already run and saved problem would be used to show, if False the algorithms would run and the show the results.
    SAVE_RESULTS: if True the rersults of already run problems would be saved, this one is highly recommended to be set True. 

- For NPG algorithms if you want to tune multiple parameters set at the same time, you need to follow the instruction commented in each test problem's file.

- Initial stepsize is set to be using linesearch with problem lasso, min_length_curve and NMF, and it's fixed at 0.001 for problems maximum_likelyhood, dual_max_entropy.

- We use numpy.random.seed as the mechanism to create re-producible results, our datasets 1-10 correspond to random seed 1-10. Problem with same size and random seed should always produce identical results.

- To produce the results in the paper you just simply run the code without any changes, we've already pre-set up the size and the random seed that we use for our paper.

- For the figures illustrated in the paper, the detailed sizes and random seeds for each problem presented in paper are provided here:
    Dual max entropy problem    : m = 4000, n = 5000, seed = 4
    Lasso                       : m = 2048, n = 8192, seed = 2
    Min length curve problem    : m = 2000, n = 10000, seed = 1
    NMF problem                 : m = 3000, n = 3000, r = 30, seed = 1
    Maximum likelyhood problem  : n = 100, lb = 0.1, ub = 10, M = 50, seed = 6
