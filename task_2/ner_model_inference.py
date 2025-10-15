from ner_model import NERmodel

entity_recognition_model_path ='./trained_ner_model'
string='There is a dog.'
model=NERmodel(entity_recognition_model_path)
animal_name,tokens = model.extract_animals(string)
print(f'\nTokens predicted: {tokens}.\n')
print(f'\nAnimal predicted : {animal_name}. ')

