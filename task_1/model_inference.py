from MnistClassifier import MnistClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
def main():
    test_size = 0.3

    data = load_digits()  # Load MNIST dataset
    X = data.data / 255
    Y = data.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size)  # Split dataset into train and test sets

    model = MnistClassifier(algorithm='rf')  # initialize one of three models: rf,cnn or nn.
    model.train(x_train, y_train)  # fit the train data
    y_pred = model.predict(x_test)  # predict test data
    acc_score = accuracy_score(y_test, y_pred)  # evaaluate metric
    print(f'Accuracy score for test data: {acc_score}')

if __name__ == "__main__":
    main()

