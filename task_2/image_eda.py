import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
train=pd.read_csv('y_Train_images.csv')
test=pd.read_csv('y_Test_images.csv')

train_counts=train.value_counts()
test_counts=test.value_counts()

image_1_path='./Dog_eda.jpeg'
image_2_path= './Tiger_eda.jpeg'
image_3_path= './VM_eda.jpeg'
image_4_path= './Panda_eda.jpeg'
images=[image_1_path,image_2_path,image_3_path,image_4_path]

fig,axes=plt.subplots(2,2)
for i,img_path in enumerate(images):
    image = Image.open(img_path)
    if i>1:
        i-=2
        axes[1, i].imshow(image)
    else:
        axes[0, i].imshow(image)



plt.show()

train=pd.read_csv('y_Train_images.csv')
test=pd.read_csv('y_Test_images.csv')

fig=plt.figure()
fig.subplots_adjust(bottom=0.25)
ax = plt.gca()
plt.ylabel('Count of appearance')
plt.xlabel('Animal name')



train_counts.plot(kind="bar", color="skyblue", alpha=0.7, ax=ax, position=0, width=0.4, label="Train")
test_counts.plot(kind="bar", color="salmon",  alpha=0.7, ax=ax, position=1, width=0.4, label="Test")
plt.xlabel('Animal name')
plt.legend()
plt.title(f"Comparison of train and test datasets' size for each animal. \nTotal number of categories: {np.unique(train).size}")
plt.show()



print('a')