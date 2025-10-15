from ner_model_trainer import NERModelTrainer

def main():
    trainer = NERModelTrainer()
    trainer.create_dataset()
    print('Dataset for training is created.')
    trainer.train()
    print('Model is exported.')

if __name__ == "__main__":
    main()
