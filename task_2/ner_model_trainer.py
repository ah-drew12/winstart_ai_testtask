import pandas as pd
from transformers import TFBertForTokenClassification,BertTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re

random_state=42
batch_size=2
train_epochs=5
learning_rate=5e-5
test_size=0.3

animals =  [
    "eagle", "robin", "fox", "hawk", "elk", "seal", "snake", "chickens", "wolf", "bison", "pigeon",
    "otter", "whale", "canary", "panther", "antelope", "hedgehog", "frog", "bee", "lynx", "buffalo",
    "coyote", "mole", "mountain goat", "sparrows", "raccoon", "rooster", "pig", "woodpecker", "falcon",
    "lemur", "crab", "skylark", "iguana", "duck", "snail", "gecko", "chipmunk", "tarantula", "osprey",
    "kingfisher", "kestrel", "ladybug", "heron", "python", "salamander", "warbler", "dragonfly",
    "caterpillar", "jaguar", "tortoise", "flamingo", "gibbon", "capybara", "gazelle", "dormouse",
    "finch", "crocodile", "snow leopard", "howler monkey", "water buffalo", "leopard", "horse",
    "gull", "chameleon", "beetle", "butterfly", "cat", "cow", "dog", "elephant", "gorilla", "hippo",
    "lizard", "monkey", "mouse", "panda", "spider", "tiger", "zebra"
] #Classes of animals

sentences=['There is a [animal] in the picture.',
           'There is a [animal] in the image.',
'There is a [animal] in the photo.'
,'A [animal] is in the picture.'
,'A [animal] is in the image.'
,'A [animal] is in the photo.'
,'The picture shows a [animal].'
,'The image shows a [animal].'
,'The photo shows a [animal].'
,'This picture contains a [animal].'
,'This image contains a [animal].'
,'This photo contains a [animal].'
,'I see a [animal] in the picture.'
,'We can see a [animal] in the image.'
,'You can see a [animal] in the photo.'
,'The picture depicts a [animal].'
,'The image depicts a [animal].',
'The photo depicts a [animal].',
'A [animal] appears in the picture',
'A [animal] appears in the image.'
           ] #Sentences sample for generation of dataset


def generate_texts(sentences, animals):
    "Method for generating the dataset. It takes the sample of sentence and replace string [animal] with animal name."
    texts = []
    y_text = []
    for sentence in sentences:
        for animal in animals:
            texts.append(sentence.replace("[animal]", animal))
            y_text.append(animal)
    return texts, y_text
def encode_texts(label):
    "Encoding tokens in sentences with ner_tags "
    labels_encoding={0:"O", 1: "B-ANIMAL"}
    return [labels_encoding[label[i]] for i in range(len(label))]

def encode_message(message, animals):
    "Encoding tokens with target as 1 where it signals that token is related to animals"
    message = message.lower()
    words = re.findall(r"\w+", message)
    encoded_words = [1 if any(animal in word for animal in animals) else 0 for word in words]
    return encoded_words



class NERModelTrainer():
    def __init__(self,random_state=random_state,test_size=test_size,batch_size=batch_size,train_epochs=train_epochs, learning_rate=learning_rate):

        #Parameters initialization
        self.random_state=random_state
        self.batch_size=batch_size
        self.train_epochs=train_epochs
        self.test_size=test_size
        self.learning_rate=learning_rate

        #Model will be based on bert-base-uncased
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def pad_to_max_length(self, sequences, max_length, padding_value=0):

        return [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]

    def tokenize_and_align_labels(self, examples, label_all_tokens=True):
        "Function tokenizes words, aligns the labels with the subtokens, and applies padding"

        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        max_length = 24

        #
        for i, (label, text) in enumerate(zip(examples["labels"], examples["text"])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            current_word = None
            label_ids = []

            # Word_idx is related to subtokens. If word_idx is None (start or end of sentence) or bigger than size of label then its label equals 0.
            # Else it returns label from array of original labels.  Each time updating max_length of label to padding arrays to the same length.
            for word_idx in word_ids:
                if word_idx != current_word:
                    current_word = word_idx
                    label_temp = 0 if (word_idx is None) or (word_idx >= len(label)) else label[word_idx]
                    label_ids.append(label_temp)

                elif word_idx is None:
                    label_ids.append(0)
                else:
                    label_ids.append(0)

            labels.append(label_ids)

            max_length = max(max_length, len(label_ids))
        #Extend arrays to the same length
        tokenized_inputs['input_ids'] = pad_sequences(tokenized_inputs['input_ids'], maxlen=max_length, padding="post",
                                                      value=0)
        labels = pad_sequences(labels, maxlen=max_length, padding="post", value=0)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def create_dataset(self):
        "Function creates the dataset, then tokenizes it and prepares train and test data with their export afterwards"

        texts,y_text=generate_texts(sentences,animals) #Generate dataset

        train_texts,test_texts,y_train,y_test=train_test_split(texts,y_text,test_size=self.test_size,random_state=self.random_state) #split into train/test data


        ########################################################################################################################################################################################################################################################################################################################

        #create labels for datasets
        train_labels = [encode_message(text, animals) for text in train_texts]
        test_labels = [encode_message(text, animals) for text in test_texts]

        #export data for EDA
        pd.DataFrame(train_texts).to_csv('train_texts.csv',index=False)
        pd.DataFrame(y_train).to_csv('train_labels.csv',index=False)

        pd.DataFrame(test_texts).to_csv('test_texts.csv',index=False)
        pd.DataFrame(y_test).to_csv('test_labels.csv',index=False)
        print('Train and test datasets are exported')

        train_texts=np.array(train_texts)
        test_texts=np.array(test_texts)

        #Create ner_tags for data
        train_ner_tags=[encode_texts(label) for label in train_labels]
        test_ner_tags=[encode_texts(label) for label in test_labels]

        #Tokenize data
        train_tokens=[self.tokenizer.tokenize(text) for i,text in enumerate(train_texts)]
        test_tokens=[self.tokenizer.tokenize(text) for i,text in enumerate(test_texts)]

        #Combine all data in dataset
        train_dataset=Dataset.from_dict({"text": train_texts, "labels": train_labels, "tokens": train_tokens,'ner_tags':train_ner_tags})
        test_dataset=Dataset.from_dict({"text": test_texts, "labels": test_labels, "tokens": test_tokens,'ner_tags':test_ner_tags})

        #Combine train and test data into one dataset
        self.dataset = DatasetDict({"train": train_dataset,
                                     "test": test_dataset})




        #Align labels to subtokens and extend arrays with padding if it needed
        tokenized_dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True)


        max_length_train = max(len(seq) for seq in tokenized_dataset["train"]["input_ids"])
        max_length_test = max(len(seq) for seq in tokenized_dataset["test"]["input_ids"])
        max_length = max(max_length_train, max_length_test)

        # Get data from tokenized dataset and extend attention mask with padding that all arrays will be the same size
        train_inputs = tokenized_dataset["train"]["input_ids"]
        train_attention_mask = self.pad_to_max_length(tokenized_dataset["train"]["attention_mask"], max_length, padding_value=0)
        self.train_labels = tokenized_dataset["train"]["labels"]

        test_inputs = tokenized_dataset["test"]["input_ids"]
        test_attention_mask = self.pad_to_max_length(tokenized_dataset["test"]["attention_mask"], max_length, padding_value=0)
        self.test_labels = tokenized_dataset["test"]["labels"]

        #Create batches of data for training and testing
        self.train_data = tf.data.Dataset.from_tensor_slices((
            {"input_ids": train_inputs, "attention_mask": train_attention_mask},
            self.train_labels
        )).batch(self.batch_size)

        self.test_data = tf.data.Dataset.from_tensor_slices((
            {"input_ids": test_inputs, "attention_mask": test_attention_mask},
            self.test_labels
        )).batch(self.batch_size)




    def train(self):
        "Function for model training"

        #Initialization of optimizer, loss function and metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #Model training
        self.model.fit(self.train_data, epochs=self.train_epochs)

        #Quality check
        predictions = self.model.predict(self.test_data)
        logits = predictions.logits #Extract logits

        predicted_labels = np.argmax(logits, axis=-1) #Round array shape from (batch_size, labels,2) to (batch_size,labels)

        y_test = np.argmax(np.array(self.test_labels), axis=-1) # Fill y_test array with position of animal label in the sentence

        metric = metrics[0] #extract metric function SparseCategoricalAccuracy
        metric.update_state(y_test, predicted_labels)

        print("Test SparseCategoricalAccuracy:", metric.result().numpy())


        #Model saving
        self.model.save_pretrained("./trained_ner_model")
        self.tokenizer.save_pretrained("./trained_ner_model")