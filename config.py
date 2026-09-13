SEED = 42

DATA_PATHS = {
    'train': 'data/train.csv',
    'test': 'data/test.csv',
}

N_SPLITS = 5


AGE_BINS = [0, 16, 32, 48, 64, 80]
FARE_QCUT_Q = 4
COMMON_TITLES = ['Mr', 'Miss', 'Mrs', 'Master']
SCALING_COLS = ['Pclass', 'SibSp', 'Parch', 'Family']
DROP_COLS = ['Name', 'Ticket', 'Age', 'Fare']
ONE_HOT_COLS = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']


C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
N_NEIGHBORS_VALUES = [3, 5, 7, 9, 11, 15, 21]
MAX_DEPTH_VALUES = [3, 5, 7, 10, None]
MIN_SAMPLES_LEAF_VALUES = [1, 5, 10, 20]
N_ESTIMATORS_VALUES = [50, 100, 200, 500]

CATBOOST_ITERATIONS_VALUES = [100, 300, 500]
CATBOOST_DEPTH_VALUES = [4, 6, 8]
CATBOOST_LR_VALUES = [0.01, 0.05, 0.1]

XGB_N_ESTIMATORS_VALUES = [100, 300, 500]
XGB_MAX_DEPTH_VALUES = [3, 5, 7]
XGB_LR_VALUES = [0.01, 0.05, 0.1]

LGBM_N_ESTIMATORS_VALUES = [100, 300, 500]
LGBM_MAX_DEPTH_VALUES = [3, 5, 7]
LGBM_LR_VALUES = [0.01, 0.05, 0.1]


DNN_ACTIVATIONS = ['relu', 'leaky_relu', 'elu']
DNN_HIDDEN_SIZES_OPTIONS = [[64], [64, 32], [128, 64, 32], [32, 16]]
DNN_DROPOUT_VALUES = [0.0, 0.2, 0.5]
DNN_BATCHNORM_OPTIONS = [False, True]
DNN_LR_VALUES = [1e-2, 1e-3, 1e-4]
DNN_BATCH_SIZE = 32
DNN_NUM_EPOCHS = 20

DNN_OPTIMIZER_VALUES = ['adam', 'adamw', 'sgd']
DNN_SCHEDULER_VALUES = [None, 'cosine']
DNN_BATCH_SIZE_VALUES = [16, 32, 64]
DNN_NUM_EPOCHS_VALUES = [20, 50]

DNN_HIDDEN_SIZES_RETEST = [[32], [64], [128]]
DNN_LR_VALUES_RETEST = [1e-2, 1e-3, 3e-3]


BEST_LGBM_PARAMS = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
BEST_DNN_MODEL_PARAMS = {'hidden_sizes': [64], 'dropout_p': 0.5, 'use_batchnorm': False, 'activation': 'leaky_relu'}
BEST_DNN_TRAIN_PARAMS = {'batch_size': 32, 'lr': 0.01, 'num_epochs': 20}
BEST_KNN_PARAMS = {'n_neighbors': 9}