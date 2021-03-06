import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import sys
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


full_data = pd.read_csv("/Users/annelouisedeboer/Documents/GitHub/classnotes2oct/src/wine_data.csv")
# looking at data
print(full_data.dtypes)
print(full_data.columns)

#second plotting figures
full_data.plot(kind="scatter", x="alcohol", y="sulphates")

data = full_data.dropna()
classifier = LabelEncoder()
full_data["class"]= classifier.fit_transform(full_data["class"])
data = full_data.dropna()
del data["Unnamed: 0"]

corr_matrix = data.corr()
print(corr_matrix["class"].apply(abs).sort_values(ascending = False))

#plotting most important attributes
attributes = ["alcohol", "volatile acidity", "citric acid", "sulphates", "density"]
pd.plotting.scatter_matrix(full_data[attributes], figsize=(12,8))

#finding groups
data_1class = data[data["class"] == 1]
data_2class = data[data["class"] == 0]
print(data.min())
poormin = data_1class.min()
poormax = data_1class.max()
goodmin = data_2class.min()
goodmax = data_2class.max()
# alcohol 8.4, volatile acidity: 0.16, citric acid: 0.0, sulphates: 0.33
#data_r = {[goodmin], [goodmax], [poormin], [poormax]}
#ranges = pd.DataFrame(data_r, columns = ["min of good", "max of good", "min of poor", "max of poor"])
# Training and Testing models
Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[["alcohol",
          "volatile acidity",
          "citric acid",
          "sulphates",
          "density"]], data["class"], random_state=30)

# training classifier
#filter for values, are there any groups in the data
# looking for groups in the data


#USING GRIDSEARCH
#Exhaustive search over specified parameter values for an estimator. Important members are fit, predict.
#GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform”
# if they are implemented in the estimator used.
# The parameters of the estimator used to apply these
# methods are optimized by cross-validated grid-search over a parameter grid.