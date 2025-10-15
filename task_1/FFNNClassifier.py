import tensorflow.python.keras
from MnistClassifierInterface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Input
import numpy as np

img_shape=64
train_epochs=50
batch_size=32
class FFNNClassifier(MnistClassifierInterface):

    """
    Feedforward Neural Network model
    """
    def __init__(self):

        input=Input(shape=(img_shape,),dtype=tf.float64)

        dense=Dense(128, activation='relu',name='dense_layer_1')(input)
        dense = Dense(256, activation='relu', name='dense_layer_2')(dense)
        dense = Dense(128, activation='relu', name='dense_layer_5')(dense)
        dense = Dense(64, activation='relu', name='dense_layer_6')(dense)
        output=Dense(10,activation='softmax',name='output_layer')(dense)

        self.model=Model(inputs=input, outputs=output)

        self.model.summary()

    def train(self,x_train,y_train,train_epochs=train_epochs,batch_size=batch_size):
        self.train_epochs = train_epochs  # Initialize train epochs parameter

        self.batch_size = batch_size  # Initialize batch size parameter

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train,epochs=self.train_epochs, batch_size=self.batch_size)
    def predict(self,x_test):
        self.model.predict(x_test)
        y_pred = self.model.predict(x_test)
        return np.argmax(y_pred, axis=1)

