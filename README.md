# Experiment on the performance of different machine learning algorithms for classification

Setup
-----
1. set up a virtual environment with minimum python version 3.8
2. install requirements in `requirements.txt` via pip
3. create directories `plots/ex1/` and `results/`

Classification
===
Scripts are located in `scripts/ex1`.
Working directory should be `scripts/ex1`.
Project root should be top level directory.
Minimum python version is 3.8. Run `python experiments.py` in `scripts/ex1` to run all experiments.
This might take a while. Results are printed on stdout and saved in `results/`.

## Used Data
- [Membership Woes Dataset](https://api.openml.org/d/44224)
A certain premium club boasts a large customer membership. The members pay an annual membership fee in return for using the exclusive facilities offered by this club. The fees are customized for every member's personal package. In the last few years, however, the club has been facing an issue with a lot of members cancelling their memberships. The club management plans to address this issue by proactively addressing customer grievances.
  
- [Zoo dataset (UCI)](https://doi.org/10.24432/C5R59V)
A simple database containing 17 Boolean-valued attributes.  The "type" attribute appears to be the class attribute. 

- [Breast Cancer](https://github.com/moritx/performance-experiment-machine-learning/tree/main/data)
A dataset which contains characteristics of tumours and one should predict if patient has "recurrence-events" or not
  
- [Loan Dataset](https://github.com/moritx/performance-experiment-machine-learning/tree/main/data)
Dataset contains characteristics about loans and one should predict the score of a loan application

## LICENCE
For scripts located in `scripts/ex1`: [MIT](https://github.com/moritx/performance-experiment-machine-learning/blob/main/LICENSE)

For data located in `data`: [CC-BY-NC-ND](https://github.com/moritx/performance-experiment-machine-learning/blob/main/data/LICENSE)

## Authors

Maximilian Süss
Katharina Schindegger 
Moritz Fischer

