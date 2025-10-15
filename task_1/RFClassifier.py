from MnistClassifierInterface import MnistClassifierInterface
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

kfold_split=10
class RFClassifier(MnistClassifierInterface):


    """
    Random Forest model
    """


    def __init__(self):

        self.model=RandomForestClassifier()

    def train(self,x_train,y_train,kfold_split=kfold_split):
        self.kfold_split=kfold_split

        self.model.fit(x_train, y_train)
        # y_pred=self.model.predict(x_train)#evaluate prediction on default settings
        best_accuracy_score=0
            # accuracy_score(y_train, y_pred) #calculate and set as start point the model on default settings
        # best_model
        accuracy_train_arr, accuracy_test_arr = [], []
        kf = KFold(n_splits=self.kfold_split)


        for i, (train_index, test_index) in enumerate(kf.split(x_train)):

            X_train_fold, X_test_fold = x_train[train_index],x_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]


            distributions = {
                'n_estimators': [10, 50, 100, 200, 300],
                'max_depth': [9, 15, 20, 30],
                'criterion' : ['gini','entropy'],
                'min_samples_leaf': [1,5,10,20,50,100],
                'warm_start': [True,False],
                'bootstrap': [True, False],
                'min_samples_leaf': [1,5,10,20,50,100],
                'max_leaf_nodes': [5,10,20,50,100,200],

            }

            clf = RandomizedSearchCV(self.model, distributions,
                                     verbose=0, scoring='accuracy')
            search = clf.fit(X_train_fold, y_train_fold)

            model = search.best_estimator_

            print('Search is finished. Best parameters: \n')
            print(search.best_params_)


            y_pred = model.predict(X_train_fold)
            accuracy_train = accuracy_score(y_train_fold, y_pred)

            y_pred = model.predict(X_test_fold)
            accuracy_test = accuracy_score(y_test_fold, y_pred)

            accuracy_train_arr.append(accuracy_train)
            accuracy_test_arr.append(accuracy_test)

            print(f'\n FOLD-{i}. Accuracy train: {accuracy_train}, Accuracy test: {accuracy_test} \n')


            if accuracy_test > best_accuracy_score:
                best_accuracy_score = accuracy_test
                best_model = model
                print(f' FOLD-{i}. Model has best score.')

        self.model = best_model


    def predict(self, x_test):
        return self.model.predict(x_test)


