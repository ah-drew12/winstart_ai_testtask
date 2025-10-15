from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    MnistClassifierInterface includes 2 methods for train model and prediction.
    It called by CNNClassifier, RFClassifier, FFNNClassifier
    """
    @abstractmethod
    def train(self,x_train,y_train):
        pass

    @abstractmethod
    def predict(self,x_test):
        pass

