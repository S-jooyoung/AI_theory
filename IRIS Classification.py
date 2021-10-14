# 데이터셋 준비
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

# 로지스틱회귀에 의한 훈련
log_reg = LogisticRegression(random_state=0).fit(X_train, y_train)

# 데이터 시각화 (Train Data)

y_train_pred = log_reg.predict(X_train)
correct_train_index = y_train_pred == y_train
false_train_index = y_train_pred != y_train

y_test_pred = log_reg.predict(X_test)
correct_test_index = y_test_pred == y_test
false_test_index = y_test_pred != y_test

plt.rcParams["figure.figsize"] = (12, 4)

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
           X_train[y_train == 0, 2], c="r", label='setosa')
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
           X_train[y_train == 1, 2], c="g", label='versicolor')
ax.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1],
           X_train[y_train == 2, 2], c="b", label='virginica')

plt.legend(), plt.grid(), plt.title("Iris data training set")
ax.set_xlabel("Sepal length"), ax.set_ylabel(
    "Petla length"), ax.set_zlabel("Petla width")


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(X_train[correct_train_index, 0], X_train[correct_train_index, 1],
           X_train[correct_train_index, 2], c="r", marker="o", label="Correct")
ax.scatter(X_train[false_train_index, 0], X_train[false_train_index, 1],
           X_train[false_train_index, 2], c="b", marker="x", label="False")

plt.legend(), plt.grid(), plt.title("Iris data training set")
ax.set_xlabel("Sepal length"), ax.set_ylabel(
    "Petla length"), ax.set_zlabel("Petla width")


plt.show()


# 데이터 시각화 (Test Data)

plt.rcParams["figure.figsize"] = (12, 4)

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
           X_test[y_test == 0, 2], c="r", label='setosa')
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
           X_test[y_test == 1, 2], c="g", label='versicolor')
ax.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1],
           X_test[y_test == 2, 2], c="b", label='virginica')

plt.legend(), plt.grid(), plt.title("Iris data testing set")
ax.set_xlabel("Sepal length"), ax.set_ylabel(
    "Petla length"), ax.set_zlabel("Petla width")


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(X_test[correct_test_index, 0], X_test[correct_test_index, 1],
           X_test[correct_test_index, 2], c="r", marker="o", label="Correct")
ax.scatter(X_test[false_test_index, 0], X_test[false_test_index, 1],
           X_test[false_test_index, 2], c="b", marker="x", label="False")

plt.legend(), plt.grid(), plt.title("Iris data testing set")
ax.set_xlabel("Sepal length"), ax.set_ylabel(
    "Petla length"), ax.set_zlabel("Petla width")

# 성능평가

print("Testing set performance:", log_reg.score(X_train, y_train))
print("Test set performance:", log_reg.score(X_test, y_test))
