# ML_PHW

# PHW1 (classification)
- decision tree(entropy)
- decision tree(gini)
- logistic regression
- support vector machine

findBestCombination(X, y, scaleList=None, modelList=None)

X = data

y = label

scaleList = scaler list (e.g. ["StandardScaler", "MinmaxScaler"]) or None (default)

modelList = model algorithm list (e.g. ["DecisionTreeEntropy", "LogisticRegression"]) or None (default)

Return values : accuracy, scaler, model, parameter, K(cross validation)


# 

Given the scaler and model as input values,
it returns which scaler and which parameters of which model show the best accuracy.


result

![image](https://user-images.githubusercontent.com/76082792/141659038-eeaf4037-9469-4bdb-8a45-0d4cdfa0340d.png)

======================================================================================================================================================

# PHW2 (clustering)
- K-means
- EM(GMM)
- CLARANS
- DBSCAN
- OPTICS

AutoML(X, y, scale_col, encode_col, scalers, encoders, features, feature_param, models, model_param, scores, score_param)
X = Data Feature

y = Data Target (If you have a target value, enter it)

scale_col = columns to scaled

encode_col = columns to encode

scalers = list of scalers (None: default)

encoders = list of encoders (None: default)

feature = list of features (None: [PCA(), RandomSelect(), CustomSelect()])

feature_param = feature selection method's parameter

models = list of models (None: default)

model_param = list of model's hyperparameter (None: default)

scores = list of score methods

score_param = list of score method's hyperparameter


Return values = some scores, plots
#

When parameters are put in, the plot and scores are output
The method of producing results in AutoML function consists of three main steps

Step 1 = Feature Selection (PCA(), RandomSelect(), CustomSelect()) * model (KMeans(), GMM(), clarans(), DBSCAN(), OPTICS()) = 15,
         Find a combination with the best silhouette score in each combination

Step 2 = If there is a target value, Among the three Feature Selection (PCA(), RandomSelect(), CustomSelect()),
         check which model has the highest purity and return three results

Step 3 = Using the final three combinations (without a target value),
         we compare with the combinations (with a target value)
         - The results are checked through the clustering plot and the silhouette score -



result

![image](https://user-images.githubusercontent.com/76082792/141659085-02cae8de-78e9-422b-8c88-b46f08ed4a58.png)
![image](https://user-images.githubusercontent.com/76082792/141659080-605b0f85-12a8-4b1f-a720-a7a6c6836fe4.png)


