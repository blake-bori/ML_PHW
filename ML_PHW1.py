import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Function to find the best combination of scaling and learning model
def findBestCombination(X, y, scaleList=None, modelList=None):
    defaultScaleDict = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler(), "MinMaxScaler": MinMaxScaler()}

    defaultModelDict = {"DicisionTreeEntropy": {"model": DecisionTreeClassifier(criterion="entropy"),
                                         "param": {"max_depth": [None, 2, 3, 4, 5, 6],
                                                   "max_leaf_nodes": [None, 2, 3, 4, 5, 6, 7],
                                                   "min_samples_split": [2, 3, 4, 5, 6],
                                                   "min_samples_leaf": [1, 2, 3],
                                                   "max_features": [None, "sqrt", "log2", 3, 4, 5]}},
                 "DecisionTreeGini": {"model": DecisionTreeClassifier(criterion="gini"),
                                      "param": {"max_depth": [None, 2, 3, 4, 5, 6],
                                                "max_leaf_nodes": [None, 2, 3, 4, 5, 6, 7],
                                                "min_samples_split": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 2, 3],
                                                "max_features": [None, "sqrt", "log2", 3, 4, 5]}},
                 "LogisticRegression": {"model": LogisticRegression(),
                                        "param": {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                                                  "C": [0.01, 0.1, 1, 10],
                                                  "max_iter": [2000]}},
                 "SVM": {"model": SVC(),
                         "param": {"C": [0.1, 1, 10, 100],
                                   "kernel": ["rbf", "poly", "sigmoid", "linear"],
                                   "degree": [1, 2, 3, 4, 5],
                                   "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}}}

    if(scaleList == None):
        scaleList=defaultScaleDict

    if(modelList == None):
        modelList=defaultModelDict


    bestScore = 0
    bestScaler = None
    bestModel = None
    bestParam = {}
    bestKFold = 0

    # Loop as number of the scaler list
    for scale in defaultScaleDict:
        if scale not in scaleList:
            continue

        scaledX = defaultScaleDict[scale].fit_transform(X)

        # Loop as number of model list
        for model in defaultModelDict:
            if model not in modelList:
                continue

            for i in range(5, 11):
                grid_search = GridSearchCV(defaultModelDict[model]["model"], param_grid=defaultModelDict[model]["param"], cv=i,
                                           n_jobs=-1)
                grid_search.fit(scaledX, y)

                curScore = grid_search.best_score_
                curModel = model
                curParam = grid_search.best_params_
                curScaler = scale
                curKFold = i

                if (curScore > bestScore):
                    bestScore = curScore
                    bestModel = curModel
                    bestParam = curParam
                    bestScaler = curScaler
                    bestKFold = curKFold

    return bestScore, bestScaler, bestModel, bestParam, bestKFold


# Read dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
           'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

df = pd.read_csv(url, header=None, names=columns)

# preprocessing
# 1. Erase the id column of the data
# 2. Change the value of the ? in the data to an appropriate value
# 3. divide the data into feature and target
df.drop("ID", axis=1)
df = df.replace('?', np.NaN)
df.fillna(method="ffill", inplace=True)

y = df["Class"]
X = df.drop('Class', axis=1)

# Put the data, scaler list, and model list into the function parameter
bestScore, bestScaler, bestModel, bestParam, bestKFold = findBestCombination(X, y,["StandardScaler","RobustScaler"],"SVM")

# Scaler list and model list are optional, soo you can just put in the data
bestScore, bestScaler, bestModel, bestParam, bestKFold = findBestCombination(X, y)

# Print the best combination and the results
print("Best Combination")
print("Best Scaler : ", bestScaler)
print("Best Model : ", bestModel)
print("Best Parameter : ", bestParam)
print("Best KFold : ", bestKFold)
print("Accuracy : ", bestScore)
