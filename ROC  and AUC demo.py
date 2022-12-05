# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:32:55 2022

@author: smkon
"""

 Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset=pd.read_csv(r'C:\Users\smkon\Desktop\data science\october month total class\29-10-2022 class\28th,31st\Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=5, min_samples_split=2, min_samples_leaf=1,)
#classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance

from sklearn.metrics import roc_curve, roc_auc_score, classification_report

print(classification_report(y_test,y_pred))

y_pred_prob=classifier.predict_proba(X_test)[::,1]

fpr, tpr, _ =roc_curve(y_test, y_pred_prob)

from sklearn.metrics import roc_curve, roc_auc_score, classification_report

print(classification_report(y_test,y_pred))

y_pred_prob=classifier.predict_proba(X_test)[::,1]

fpr, tpr, _=roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr,color='orange', label='Decision Tree Classification')

plt.plot([0,1],[0,1],'--')

plt.legend()

plt.title('ROC CURVE')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

from sklearn import metrics

auc = metrics.roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr,tpr,label="AUC="+str(auc))

plt.title('ROC CURVE')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend()

plt.show()

