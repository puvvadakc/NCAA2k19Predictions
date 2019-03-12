import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import BayesianRidge


def KnnFunc(train_features, train_outcome):
    """This function takes in a set of train features and outcomes and runs a Kneighbors Regression through a pipelines
    and grid search through multiple parameters of kneighbors such as number of neighors (1-40), weights, and algorithms. 
    Pipeline run with Imputer to fill in missing values and SelectKBest as the feature selection method. The function returns
    the fitted most optimal prediction model from the grid search conducted."""
    scaler = MinMaxScaler()
    imputer = Imputer()
    knn = KNeighborsRegressor()
    param_grid = {'kneighborsregressor__n_neighbors': range(1,40),
                 'kneighborsregressor__weights': ['uniform', 'distance'],
                 'kneighborsregressor__algorithm' :['kd_tree', 'ball_tree', 'brute']}
    pipe = make_pipeline(imputer, scaler, SelectKBest(), knn)
    grid_search = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")
    grid_search.fit(train_features, train_outcome)

    return grid_search


# Decision Tree Regression
def DecisionTreeFunc(train_features, train_outcome):
    """This function takes in a set of train features and outcomes and runs a Decision Tree Regression through a pipelines
    and grid search through max features parameter from 1-10. Pipeline run with Imputer to fill in missing values and
    SelectKBest as the feature selection method. The function returns the fitted most optimal prediction model from the
    grid search conducted."""
    tree = DecisionTreeRegressor()
    imputer = Imputer()
    param_grid = {}
    pipe = make_pipeline(imputer, SelectKBest(), tree)
    grid = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")
    grid.fit(train_features, train_outcome)
    return grid

# Neural Network Regression
def NeuralNetworkFunc(train_features, train_outcome):
    """This function takes in a set of train features and outcomes and runs a MLP Neural Network Regression through a pipelines
    and grid search of multiple variations. MLP Regression is done with a MinMaxScaler to scale data. Pipeline run with Imputer
    to fill in missing values and SelectKBest as the feature selection method. The function returns the fitted most optimal 
    prediction model from the grid search conducted."""
    clf = MLPRegressor()
    imputer = Imputer()
    param_grid = {}
    pipe = make_pipeline(imputer, MinMaxScaler(), SelectKBest(), clf)
    grid = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")
    grid.fit(train_features, train_outcome)
    return grid

# Bayesian Ridge Regression
def BayesianRidgeFunc(train_features, train_outcome):
    """This function takes in a set of train features and outcomes and runs a Bayesian Ridge Regression through a pipelines
    and grid search of multiple variations. Pipeline run with Imputer to fill in missing values and SelectKBest as the feature 
    selection method. The function returns the fitted most optimal prediction model from the grid search conducted."""
    clf = BayesianRidge()
    imputer = Imputer()
    param_grid = {}
    pipe = make_pipeline(imputer, MinMaxScaler(),SelectKBest(), clf)
    grid = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")
    grid.fit(train_features, train_outcome)
    return grid