from sklearn.multioutput import MultiOutputRegressor 
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from scipy.stats import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import sys

print('done loading')
sys.stdout.flush()

#training adapted from https://www.kaggle.com/yassinealouini/hyperopt-the-xgboost-model

DanQ_train = np.load('../train_val_test/final_features/evolANDepi_DanQtrain_redo.npy')

allscores = pd.read_csv('../train_val_test/zscores/mintrain_zscores.csv', sep="\t", header=None)
scores = allscores.values
scores_train = scores.astype(float)

DanQ_val = np.load('../train_val_test/final_features/evolANDepi_DanQval_redo.npy')

allscores = pd.read_csv('../train_val_test/zscores/val_zscores.csv', sep="\t", header=None)
scores = allscores.values
scores_val = scores.astype(float)

#incorporating hyperopt

def score(params):
    params['n_estimators'] = int(params['n_estimators'])
    print("Training with params: ")
    print(params)
    sys.stdout.flush()

    gbm_model = MultiOutputRegressor(XGBRegressor(**params))
    gbm_model.fit(DanQ_train, scores_train)

    predictions = gbm_model.predict(kmer_val)

    #getting score, MSE
    total_se = (scores_val - predictions) ** 2
    mse = []
    for i in range(4):
        mse.append(np.mean(total_se[:, i]))    
    score = np.mean(mse)
    print("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}

def optimize(trials, random_state=321):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.3)),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(3, 40, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 50, dtype=int)),
        'subsample': hp.quniform('subsample', 0.5, 1.0, 0.05),
        'gamma': hp.loguniform('gamma', np.log(0.01), np.log(10)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.05),
        'lambda': hp.quniform('lambda', 0, 1.0, 0.05),
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo = tpe.suggest, trials = trials, max_evals = 25) #more evals
    return best

trials = Trials() #store history of search
best = optimize(trials)
print("The best hyperparameters are: ", "\n")
print(best)
sys.stdout.flush()



#add here: train final model
print('training model')

rf1 = MultiOutputRegressor(XGBRegressor(max_depth = best['max_depth'], n_estimators = int(best['n_estimators']), random_state = 123, n_jobs=-1, silent=False, 
			   colsample_bytree = best['colsample_bytree'], gamma = best['gamma'], reg_lambda = best['lambda'], learning_rate = best['learning_rate'], 
			   min_child_weight = best['min_child_weight'], subsample = best['subsample']))

rf1.fit(DanQ_train, scores_train)

joblib.dump(rf1, './new_xg.pkl')
