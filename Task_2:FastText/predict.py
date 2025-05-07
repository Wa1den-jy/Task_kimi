import fasttext
from datasets import load_dataset
from config import Config
import warnings

def predict_fineweb():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    model = fasttext.load_model(Config.MODEL_PATH)
    
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    samples = []
    for i, data in enumerate(dataset):
        if i >= Config.TEST_SIZE:
            break
        text = data["text"].replace("\n", " ").strip()
        samples.append(text)
    
    with open(Config.PREDICT_OUTPUT_PATH, "w") as f:
        for text in samples:

            prediction = model.predict(text)
            label = prediction[0][0] 
            f.write(f"{label} {text}\n")
    
    print(f"预测完成！结果已保存至 {Config.PREDICT_OUTPUT_PATH}")

if __name__ == "__main__":
    predict_fineweb()