# Configuration file - all parameters in one place
RANDOM_STATE = 42
K_RANGE = range(2, 11)
PCA_COMPONENTS = 2
DBSCAN_EPS = 1.2
DBSCAN_MIN_SAMPLES = 10

FEATURES_BASE = [
    'H Shots', 'A Shots', 'H SOT', 'A SOT',
    'H Fouls', 'A Fouls', 'H Yellow', 'A Yellow',
    'H Red', 'A Red', 'H Corners', 'A Corners'
]

TRAIN_TEST_SPLIT = 0.2
RF_N_ESTIMATORS = 400

# Paths
DATA_PATH = "data/england.csv"
RESULTS_PATH = "results/"
