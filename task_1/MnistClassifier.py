
from CNNClassifier import CNNClassifier
from RFClassifier import RFClassifier
from FFNNClassifier import FFNNClassifier



class MnistClassifier:
    """
    MnistClassifier includes 3 algorithms for classification depending on attribute "algorithm".

    algorithm: "rf","nn" ,"cnn"
    """

    def __init__(self, algorithm=None):

        if algorithm is None:
            raise ValueError("No model was specified in attribute 'algorithm'. Specify this attribute with one of these: 'rf', 'fnn', 'cnn'.")
        if algorithm == "cnn":
            print('Convolution Neural Network model is called')
            self.model = CNNClassifier()
        elif algorithm == "rf":
            print('Random Forest model is called')
            self.model = RFClassifier()

        elif algorithm == "nn":
            print('Feedforward Neural Network model is called')
            self.model = FFNNClassifier()
        else:
            raise ValueError("Wrong algorithm was called. Use 'cnn', 'rf', or 'nn'.")

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
