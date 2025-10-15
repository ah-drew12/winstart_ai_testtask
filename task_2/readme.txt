Hello, in this readme file i will tell you how to run the notebook on your device.

1. The first thing to do is to install all libraries from requirements.txt. 


2. After that the notebook tells us about image classification model that was trained on dataset (https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset).
You need to extract folders "Training Data" and "Testing Data" to the directory with notebook. After that you should rename it to train_data and test_data respectively.
 ** Make sure that after you open folder train_data you instantly see folders with animals name not a second folder with the same name ("Training Data" or "Testing Data").
When you renamed folders, you can run the part of notebook with image classification model training. When training will be finished, in this folder should appear file with model weights which is called 'image_classification_model.h5'. You will use this file in the next parts to inference model.

3. Images for inference of image classification should be downloaded with other scripts.

4. The fourth part of notebook is to finetune 'bert-base-uncased' model that should be downloaded automatically (when you call it ensure the transformers library is installed — it is already included in requirements.txt).
To run finetuning process you should download any dataset because sentences will be generated from samples in scripts. It has list of animals that will be filled into sample and generating dataset. After you finished the finetuning the parameters of model will be saved into the folder named 'trained_ner_model'. From this folder you will load trained NERmodel in future.

5. Inference of bert model will not require any data to download. There are sentences in script already.

6. Pipeline creation consist of loading weights of models that you trained before. Image and sentences for inference pipeline should be downloaded with other scripts.


Good luck deploying and experimenting with this notebook
