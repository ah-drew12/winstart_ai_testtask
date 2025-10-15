
import numpy as np
from pipeline import Pipeline_for_task_2
import tensorflow as tf
import PIL
from ner_model import NERmodel
from PIL import Image
import matplotlib.pyplot as plt

path_to_image1 ='./Dog-Test (46).jpeg'
path_to_image2 ='./Elephant-Test (380).jpeg'
path_to_image3 ='./Cow-Test (15).jpeg'

string= "The Unicorn pranced under the rainbow."

text1 ="My dog loves to sleep on the couch."
text2 ="An elephant walked through the clearing."
text3 ="The cow drank from the pond."

data =[(path_to_image1 ,text1) ,(path_to_image2 ,text2) ,(path_to_image3 ,text3)]

classification_model_path ='image_classification_model.h5'
entity_recognition_model_path ='./trained_ner_model'


pipeline =Pipeline_for_task_2(classification_model_path ,entity_recognition_model_path)


for img ,text in data:
    image =np.asarray(Image.open(fr"{img}").convert('L').resize((64 ,64)))
    image = np.expand_dims(image, -1)
    # plt.imshow(np.squeeze(image))
    plt.title((text))
    plt.show()
    image = image.reshape((-1, *image.shape ))

    result =pipeline.evaluate(image ,text)

    print(f'Result of pipeline: {result}')

