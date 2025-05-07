import fasttext
from config import Config

def train_model():
    model = fasttext.train_supervised(
        input=Config.TRAIN_DATA_PATH,
        epoch=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        wordNgrams=Config.WORD_NGRAMS,
        dim=Config.DIM,
        loss=Config.LOSS
    )
    
    model.save_model(Config.MODEL_PATH)
    
    result = model.test(Config.VALID_DATA_PATH)
    print(f"Validation Results:")
    print(f"Number of examples: {result[0]}")
    print(f"Accuracy: {result[1]*100:.2f}%")

if __name__ == "__main__":
    train_model()