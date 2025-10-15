import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Input, Conv2D,MaxPooling2D,Conv3D,MaxPooling3D
from tensorflow.keras import metrics as keras_metrics
import numpy as np
from keras.models import load_model
from PIL import Image


epochs=10
batch_size=32
resized_img_shape=(64,64)
animals = ['beetle', 'butterfly', 'cat', 'cow', 'dog', 'elephant', 'gorilla', 'hippo', 'lizard', 'monkey',
           'mouse', 'panda', 'spider', 'tiger', 'zebra']  # animals on which the image classification model was trained

class CNNClassifier:
    """
    Convolution Neural Network model for Animal Images Classification
    """

    def __init__(self,path=None):
        'As input receives an argument of path that indicates to load weights of the model for its inference. '
        if path is None:

            #Model architecture consist of 2 Convolution and Pooling layers combined one by one, then it flatten tensors and go through  feed forward layers
            input = Input(shape=(*resized_img_shape,1), dtype=tf.float32)

            conv1 = Conv2D(32, (3, 3), activation='relu', name='conv_layer_1')(input)
            pool1 = MaxPooling2D((2, 2), name='maxpool_layer_1')(conv1)
            conv2 = Conv2D(64, (3, 3), activation='relu', name='conv_layer_2')(pool1)
            pool2 = MaxPooling2D((2, 2), name='maxpool_layer_2')(conv2)

            flatten = Flatten()(pool2)
            dense1 = Dense(512, activation='relu', name='dense_layer_1')(flatten)
            dense2 = Dense(256, activation='relu', name='dense_layer_2')(dense1)
            dense3 = Dense(128, activation='relu', name='dense_layer_3')(dense2)
            output_layer = Dense(15, activation='softmax', name='output_layer')(dense3)
            self.model = Model(inputs=input, outputs=output_layer)
            self.model.summary()
        else:
            self.model=load_model('image_classification_model.h5')
    def train(self,x_train,y_train,epochs=epochs,batch_size=batch_size):
        'Method for model training receives as arguments X and Y, epochs and batch_size '
        self.epochs=epochs
        self.batch_size=batch_size


        x_train=x_train.reshape((x_train.shape[0],*resized_img_shape,1))
        metrics=['accuracy']
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics)
        self.model.fit(x_train, y_train,epochs=self.epochs, batch_size=self.batch_size)

    def predict(self,x_test):
        'Method for predicting the label of the animal class on image'
        #If we works with batch where number of images is large than 1 then we use tf.image.resize method. Otherwise, we use method from Pillow library.
        if x_test.shape[0]>1:
            x_test = tf.image.resize(x_test,resized_img_shape)
        else:
            img = Image.fromarray(x_test)  # turn x_test from ndarray to pil.image
            img = img.resize(resized_img_shape) #resize to needed shape
            x_test = np.asarray(img)#turn back to ndarray
            x_test = x_test[None, ..., None] #add dimension of batch and channel
        y_pred = self.model.predict(x_test)
        return np.argmax(y_pred, axis=1)
    def predict_animal_name(self,x_test):
        'Method similar as predict but returns the animal name'
        if x_test.shape[0]>1:
            x_test = tf.image.resize(x_test, resized_img_shape)
        else:
            img = Image.fromarray(x_test[0,:,:,0])  
            img = img.resize(resized_img_shape)
            x_test = np.asarray(img)
            x_test = x_test[None, ..., None]
        y_pred = self.model.predict(x_test)
        return animals[np.argmax(y_pred, axis=1)[0]]

    def save(self,path):
        "Method for saving the model's weights"
        self.model.save(path)


