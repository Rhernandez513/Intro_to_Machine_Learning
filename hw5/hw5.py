import pickle
import util
import numpy as np

# features = x
# labels = y

cv = 2

with open("vox_data.pkl", "rb") as file:
    X0, Y0, article_ids, article_links = pickle.load(file)
print("X: " + str(X0))
print("Y: " + str(Y0))
X, Y, Xte, Yte = util.splitTrainTest(X0, Y0, 5)

print("Using " + str((len(Xte) / len(X0))*100) + " Percent Test Data")


# Multi-layer Perceptron: A Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#param_grid = {'solver': ['lbfgs'],
#              'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
#              'alpha': 10.0 ** -np.arange(1, 10),
#              'hidden_layer_sizes':np.arange(1, 10),
#              'random_state':[0,1,2,3,4,5,6,7,8,9]}
#             }
#mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

param_grid = {
    'solver': ['lbfgs'],
    'alpha': 10.0 ** -np.arange(1,3),
    'hidden_layer_sizes': np.arange(1,3),
    'random_state':[0,1,2]
}

mlp_clf = GridSearchCV(MLPClassifier(), param_grid, n_jobs=-1)
mlp_clf.fit(X, Y)

print("Score: " + str(mlp_clf.score(X, Y)))
print("Score: " + str(mlp_clf.best_score_))
print("Best Params: " + str(mlp_clf.best_params_))

mlp_clf.predict(Xte)




# Random Forest: Ensemble learning by using many decision trees then taking the mean
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators' : [40, 70, 100],
    'max_depth' : [5, 15, 20],
    'max_leaf_nodes' : [5, 10, 20],
    'random_state':[0,1,2,3,4,5,6,7,8,9]
}

#rnd_clf = RandomForestClassifier(max_depth=2, max_leaf_nodes=16, random_state=0, n_jobs=-1)

rnd_clf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1)

rnd_clf.fit(X, Y)
print("Score: " + str(rnd_clf.score(X, Y)))
print("Best Score: " + str(rnd_clf.best_score_))
print("Best Params: " + str(rnd_clf.best_params_))
rnd_clf.predict(Xte)


# XGBoost: Regularized Gradient Boosting

import xgboost.sklearn as xgb
from sklearn.model_selection import TimeSeriesSplit

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), x, y, scoring='neg_mean_squared_error')

# root_mean_squared_error = (-scores)**0.5
# print(str(scores.mean()))


param_grid = {"subsample" : [0.5, 0.8]}

fit_params = {"early_stopping_rounds" : 42,
              "eval_metric" : "error",
              "eval_set" : [[Xte, Yte]]}

xgb_clf = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False),
                           param_grid, cv=TimeSeriesSplit(n_splits=cv).get_n_splits([X, Y]))

xgb_clf.fit(X, Y, **fit_params)

print("Score: " + str(xgb_clf.score(X, Y)))
print("Best Params: " + str(xgb_clf.best_params_))

xgb_clf.predict(Xte)

# Support Vector Machines

from sklearn import svm

param_grid = {'C': [1, 5, 10], 'kernel': ['linear']}

svm_clf = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1)

svm_clf.fit(X, Y)

print("Score: " + str(svm_clf.score(X, Y)))
print("Best Params: " + str(svm_clf.best_params_))

svm_clf.predict(Xte)

from sklearn.metrics import accuracy_score

mlp_y_pred = mlp_clf.predict(Xte)
rnd_y_pred = rnd_clf.predict(Xte)
xgb_y_pred = xgb_clf.predict(Xte)
svm_y_pred = svm_clf.predict(Xte)

mlp_accuracy = accuracy_score(Yte, mlp_y_pred, normalize=False) / float(Yte.size)
rnd_accuracy = accuracy_score(Yte, rnd_y_pred, normalize=False) / float(Yte.size)
xgb_accuracy = accuracy_score(Yte, xgb_y_pred, normalize=False) / float(Yte.size)
svm_accuracy = accuracy_score(Yte, svm_y_pred, normalize=False) / float(Yte.size)

#mlp_error_rate = 1 - mlp_accuracy
#rnd_error_rate = 1 - rnd_accuracy
#xgb_error_rate = 1 - xgb_accuracy
#svm_error_rate = 1 - svm_accuracy

#print("MLP Error Rate: " + str(mlp_error_rate))
#print("RND Error Rate: " + str(rnd_error_rate))
#print("XGB Error Rate: " + str(xgb_error_rate))
#print("SVM Error Rate: " + str(svm_error_rate))

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

print("MLP Accuracy: " + str(mlp_accuracy))
print("RND Accuracy: " + str(rnd_accuracy))
print("XGB Accuracy: " + str(xgb_accuracy))
print("SVM Accuracy: " + str(svm_accuracy))

ascending = sorted([mlp_accuracy, rnd_accuracy, xgb_accuracy, svm_accuracy])
print("Best accuracy value: " + str(ascending[-1]))
print("Best accuracy name: " + str(namestr(ascending[-1], globals())))

# EOF