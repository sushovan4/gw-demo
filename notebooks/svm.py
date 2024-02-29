# %% 
#  SVM - Support Vector Machines
#  SVC - Support Vector Classifier
# (SVR - Support Vector Regressor)
# Concepts: 
# Loss function
# Linear SVC  ( SVC(kernel='linear')   or linearSVC  )
# classifier that are linear separable
# (non-linear) SVC
# transformed features might be linearly separable
# Kernel SVM
# RBF  Radial Basis Function
# gamma small - smooth. (default=1)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from svm_plot import plot as plot_svm

# SVM - Support Vector Machines
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# %%
# Generate random data
rng = np.random.default_rng(1)
X = rng.standard_normal((50, 2))
y = np.array([-1] * 25 + [1] * 25)
X[y==1] += 1
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:, 0], X[:, 1], c = y, cmap=plt.cm.coolwarm)

# %%
# Support Vector Classifier
svm_linear = SVC(C = 100, kernel = 'linear')
svm_linear.fit(X, y)

# %%
# Plot SVM decision boundary
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, svm_linear, ax = ax)

# Try other values of C, e.g., C = 0.1

# %%
svm_linear.coef_
svm_linear.intercept_
# Can you interpret it?


# %%
kfold = KFold(5, random_state = 0, shuffle = True)
grid = GridSearchCV(svm_linear, 
                    {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}, 
                    cv = kfold, scoring = 'accuracy')
grid.fit(X, y)

grid.best_params_
grid.cv_results_[('mean_test_score')]


# %%
# Test data set
X_test = rng.standard_normal((20, 2))
y_test = np.array([-1] * 10 + [1] * 10)
X_test[y_test == 1] += 1

# %%
# Best estimator
best_ = grid.best_estimator_
y_hat = best_.predict(X_test)
best_.score(X_test, y_test)


# %%
# Activity: try to separate the classes further, say by 2.
# Find the best C


# %%
# Non-linear boundary
X = rng.standard_normal((200, 2))
y = np.array([1] * 150 + [2] * 50)
X[:100] += 2
X[100:150] -= 2

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:, 0], X[:, 1], c = y, cmap=plt.cm.coolwarm)

svm_rbf = SVC(kernel = 'rbf', gamma = 1, C = 1) # RBF = Radial Basis Function
svm_rbf.fit(X, y)

fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, svm_rbf, ax = ax, scatter_cmap=plt.cm.coolwarm)


# Best is probabily C = 100 and gamma = 1



# %%
# Example, hand written digits recognition
from sklearn.datasets import load_digits
digits = load_digits()
# 1797 sample figures, 8x8 pixel of 16 gray-scale digitized image
print(f'type(digits) = { type(digits) }')
print(f'The attributes/keys of digits are = { digits.__dir__() }')
#
print(f'\ndigits.data.shape = {digits.data.shape}')
print(f'Sample digits.data[0,12] = {digits.data[0,12]}')
print(f'Sample digits.data[0,13] = {digits.data[0,13]}')
#
print(f'\ndigits.images.shape = {digits.images.shape}')
print(f'Sample digits.image[0,1,4] = {digits.images[0,1,4]}')
print(f'Sample digits.image[0,1,5] = {digits.images[0,1,5]}')


# %%
# What do they look like?
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[0]) 
plt.show() 

plt.gray() 
plt.matshow(digits.images[-1]) 
plt.show() 



# %%
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)


# %% 
# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(f'lr train score:  {lr.score(X_train,y_train)}')
print(f'lr test score:  {lr.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, lr.predict(X_test)))



# %%
# Apply SVM and print scores
svc = SVC()
svc.fit(X_train,y_train)
print(f'svc train score:  {svc.score(X_train,y_train)}')
print(f'svc test score:  {svc.score(X_test,y_test)}')
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))





# %%
# Instantiate dtree
dtree_admit1 = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree_admit1.fit(X_train,y_train)
# Predict test set labels
y_test_pred = dtree_admit1.predict(X_test)
# Evaluate test-set accuracy
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))



#%%