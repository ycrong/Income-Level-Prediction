import os
os.chdir('C:\\Users\\yuche\\OneDrive\\Desktop')
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import label_binarize
from sklearn import tree,grid_search,metrics
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score, train_test_split,LeaveOneOut
from sklearn.metrics import confusion_matrix,average_precision_score,f1_score, precision_score,recall_score,classification_report,auc
from sklearn.cross_validation import KFold
from scipy.stats import sem
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pydot
from sklearn.tree import export_graphviz 

df = pd.read_csv('census_income.csv')
df = df.replace(' ?',np.nan)
df = df.dropna(how = 'any')
raw_X = df[df.columns[0:14]]
X = pd.concat([pd.get_dummies(raw_X[['workclass','education',\
                                    'marital-status','occupation','relationship','race','sex','native-country']]), \
              raw_X[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']]],ignore_index=True, axis = 1)
df.salary_level = df.salary_level.map({' <=50K':0,' >50K':1})
Y = df[['salary_level']]

#Logistic Regression
logistic = LogisticRegression()
lgt = logistic.fit(X,Y)
y_pred_lgt = lgt.fit(X, Y).predict(X)
score_l = lgt.score(X, Y)
print 'Logistic Regression Result'
print 'Decision Tree result'
print 'Accuracy:', score_l
lgt_matrix = confusion_matrix(Y, y_pred_lgt)
print 'Confusion matrix:\n',lgt_matrix
print 'f-measure:',(f1_score(Y,y_pred_lgt,average = None))
print 'precision:',(precision_score(Y,y_pred_lgt,average = None))
print 'recall:',(recall_score(Y ,y_pred_lgt,average = None))
#cross validation
lgt_classifier = LogisticRegression()
lgt_classifier.fit(X,Y)
crossvalidation = KFold(n=X.shape[0], n_folds=5,
shuffle=True, random_state=1)
score1_lgt = cross_val_score(lgt_classifier, X, Y, scoring = 'accuracy', cv=crossvalidation, n_jobs=1)
print 'Accuracy from cross validation: \n',score1_lgt
print 'Average accuracy:',np.mean(score1_lgt)

#Decision Tree
#overall accuracy
clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10, min_samples_leaf = 5, max_depth = 5)
clf = clf.fit(X, Y)
y_pred = clf.fit(X, Y).predict(X)
score = clf.score(X, Y)
print 'Decision Tree result'
print 'Accuracy:', score
clf_matrix = confusion_matrix(Y, y_pred)
print 'Confusion matrix:\n',clf_matrix
print 'f-measure:',(f1_score(Y,y_pred,average = None))
print 'precision:',(precision_score(Y,y_pred,average = None))
print 'recall:',(recall_score(Y ,y_pred,average = None))

#Cross Validation
tree_classifier = tree.DecisionTreeClassifier(
  min_samples_split=30, min_samples_leaf=10,
  random_state=0)
tree_classifier.fit(X,Y)
crossvalidation = KFold(n=X.shape[0], n_folds=5,
  shuffle=True, random_state=1)
score1 = cross_val_score(tree_classifier, X, Y, scoring = 'accuracy', cv=crossvalidation, n_jobs=1)
print 'Accuracy from cross validation: \n',score1
print 'Average accuracy:',np.mean(score1)

#Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

clf = RandomForestClassifier(n_estimators=57, max_depth=None,
      min_samples_split=2, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.fit(X_train,y_train).predict(X_test)
score = clf.score(X_test, y_test)

crossvalidation = KFold(n=X_train.shape[0], n_folds=10,
shuffle=True, random_state=1)
score1 = cross_val_score(clf, X, Y, scoring = 'accuracy', cv=crossvalidation, n_jobs=1) 
score1.mean()

print 'Random Forest Result'
print 'Accuracy:', score
knn_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',knn_matrix
print 'f-measure:',(f1_score(y_test,y_pred,average = None))
print 'precision:',(precision_score(y_test,y_pred,average = None))
print 'recall:',(recall_score(y_test,y_pred,average = None))
print 'Accuracy from cross validation: \n',score1
print 'Average accuracy:',np.mean(score1)