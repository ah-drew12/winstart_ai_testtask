import pandas as pd
import tensorflow as tf
import numpy as np
from image_classification_model import CNNClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

train_path='train_data/'
test_path='test_path/'
max_count=300 #Limit on the number of photos
img_channels=1 #number of img channels
image_size=(64,64) #size to which it should resize imported images
X = []
Y = []

train_epochs=10
train_batch_size=32
#Load images, normalize and resize them
def load_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=img_channels)
    img=tf.cast(img,tf.float32)/255
    img=tf.image.resize(img,image_size)
    return img

def import_folder(path,max_count=100):
    x_data,y_data=[],[]
    for animal_folder in os.listdir(train_path):
        count=0
        for image_path in os.listdir(os.path.join(train_path,animal_folder)):
            if count>max_count: #import N images from the folder. It limits the size of dataset
                break
            x_data.append(load_img(os.path.join(train_path,animal_folder,image_path)))
            y_data.append(str.lower(animal_folder))
            count+=1

    return np.array(x_data),np.array(y_data)


#Import images
x_train,y_train=import_folder(train_path,max_count)

x_test,y_test=import_folder(test_path)

pd.DataFrame(y_train).to_csv('y_Train_images.csv',index=False)
pd.DataFrame(y_test).to_csv('y_Test_images.csv',index=False)
print('Importing is finished.')

#Encode Y labels of images into indexes of animal names
labelencoder=LabelEncoder()
y_train=labelencoder.fit_transform(y_train)
y_test=labelencoder.transform(y_test)
print('LabelEncoding is finished.')
y_train = y_train.reshape(-1,)

#Initilialization of  model
model=CNNClassifier()


#Model training
print('\nStart training.')
model.train(x_train,y_train,epochs=train_epochs,batch_size=train_batch_size)

#Model predict and quality assessment
y_pred=model.predict(x_test)

acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy score for test data: {acc_score}')


model.save('image_classification_model.h5')

