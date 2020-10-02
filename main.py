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


data_dir = Path("/Users/annelouisedeboer/PycharmProjects/classnotes")
full_data = pd.read_csv(data_dir / "wine_data.csv")
# looking at data
print(full_data.dtypes)
print(full_data.columns)

#second plotting figures
full_data.plot(kind="scatter", x="alcohol", y="sulphates")


#data = data.dropna()
classifier = LabelEncoder()
full_data["class"]= classifier.fit_transform(full_data["class"])
data = full_data.dropna()
del data["Unnamed: 0"]

corr_matrix = data.corr()
print(corr_matrix["class"].apply(abs).sort_values(ascending = False))

#plotting most important attributes
attributes = ["alcohol", "volatile acidity", "citric acid", "sulphates", "density"]
pd.plotting.scatter_matrix(full_data[attributes], figsize=(12,8))

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