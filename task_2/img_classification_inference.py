from image_classification_model import CNNClassifier
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create variable with paths to images
path_to_image1 ='./Dog-Test (46).jpeg'
path_to_image2 ='./Elephant-Test (380).jpeg'
path_to_image3 ='./Cow-Test (15).jpeg'

#Path to the weights of model
classification_model_path ='image_classification_model.h5'

#Combine variable into one array
data =[path_to_image1 ,path_to_image2 ,path_to_image3]

#Load the model
model=CNNClassifier(classification_model_path)

for img  in data:
    image =np.asarray(Image.open(fr"{img}").convert('L'))#Import image
    image = np.expand_dims(image, -1)
    # plt.imshow(image, cmap="gray")
    plt.show()
    image=image.reshape(-1,*image.shape)



    result=model.predict_animal_name(image)

    print(f'Model classify image as the photo of : {result}')
