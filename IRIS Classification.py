# 데이터셋 준비
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, train_size=0.7)

X_train = X_train[:, [0, 2, 3]]
X_test = X_test[:, [0, 2, 3]]
# Sepal length , petal length, petal width 추출
