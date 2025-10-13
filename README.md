# Time-integrated Optimal Transport — Experimental Code

This repository provides the experimental code and data used in the paper “Time-integrated Optimal Transport: A Robust Minimax Framework.” 
It includes all scripts necessary to reproduce the results and figures presented in the experimental section.

### Datasets:
 - Data for experiments in Sections 4.1 and 4.3 is synthetically generated.
 - Data supporting Section 4.2 is provided in the file DailyDelhiClimateTrain.csv.
 - Data for Section 4.4 is stored in the folder time_series_kNN.

### Code Structure:
The main solver for the TiOT and eTiOT problems, along with related components, is implemented in TiOT_lib.py.
The repository also includes five experiment scripts corresponding to the paper’s sections:
 - alignment_Exp.py – Experiment for Section 4.1 (Figure 1)
 - lag_series_Exp.py – Experiment for Section 4.2
    - run the function 'dist_lag_exp()' to reproduce Figure 2 (left)
    - run the function 'dist_w_exp()' to reproduce Figure 2 (right)
 - runtime_Exp.py – Experiment for Section 4.3
    - run the function 'deviation_experiment()' to reproduce Figure 3 (left)
    - run the function 'runtime_experiment()' to reproduce Figure 3 (right)
 - kfold_kNN_Exp.py – Generates Table 1 (Section 4.4)
 - robust_kNN_Exp.py – Generates Figure 4 (Section 4.4) and Figure 5 (Appendix D)

Running Experiments:
To reproduce results, run the desired script using:
python <script_name>.py

For example:
python lag_series_Exp.py

Experimental outputs will be saved automatically in the Experimental_outputs directory.

  
