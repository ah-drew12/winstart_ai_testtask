import numpy as np
import tensorflow as tf
from transformers import TFBertForTokenClassification, AutoTokenizer, BertConfig


class NERmodel():
    "Model class consist of 3 methods: init,train,extract_animals. Train method is implemented in separated script because of large construction. "
    def __init__(self,path_to_model):

        "Initialization of model with loading weights and tokens from given path."
        id2label = {0: "O", 1: "ANIMAL"}
        label2id = {"O": 0, "ANIMAL": 1}

        self.model=TFBertForTokenClassification.from_pretrained(path_to_model,num_labels=2,
    id2label=id2label,
    label2id=label2id)

        self.tokenizer=AutoTokenizer.from_pretrained(path_to_model)

        print('NER model is loaded.')
    def train(self,train_dataset):
        "Method of model train is in separate class"
        print('Method of ner model training is in separate class')
        pass

    def extract_animals(self,test_dataset):
        "Function for extract of animal name from sentence. eturns animal name and tokens tags"
        inputs = self.tokenizer(test_dataset, return_tensors="tf")
        output=self.model(**inputs)
        predictions =np.argmax(output.logits,axis=-1)[0]

        tokens=self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        results = [
            (token, self.model.config.id2label[label_id])
            for token, label_id in zip(tokens, predictions)
        ]

        animal_token = [token for token, tag in results if "ANIMAL" in tag]
        if len(animal_token)==0:
            animal_token=None
        else:
            animal_token=animal_token[0]


        return animal_token,results
