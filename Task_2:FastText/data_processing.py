from datasets import load_dataset
import random
from bs4 import BeautifulSoup
from config import Config
import re

def clean_text(text):
    """清洗文本：去除HTML标签、换行符，统一小写"""
    try:
        # 尝试用更宽松的解析器处理
        text = BeautifulSoup(text, "lxml").get_text()  # 使用lxml解析器
    except:
        # 如果解析失败，直接使用正则表达式移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("\n", " ").strip()  # 去除换行符
    return text.lower()  # 统一小写

def load_and_sample(dataset_name, label, sample_size):
    """从HuggingFace加载数据集并采样"""
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    samples = []
    for i, data in enumerate(dataset):
        if i >= sample_size:
            break
        text = clean_text(data["text"])
        samples.append(f"__label__{label} {text}")
    return samples

def main():
    # 加载正负样本
    math_samples = load_and_sample(
        "open-web-math/open-web-math", "math", Config.SAMPLE_SIZE
    )
    non_math_samples = load_and_sample(
        "HuggingFaceFW/fineweb", "non_math", Config.SAMPLE_SIZE
    )
    
    # 合并并打乱
    all_samples = math_samples + non_math_samples
    random.shuffle(all_samples)
    
    # 划分训练集和验证集（8:2）
    split_idx = int(0.8 * len(all_samples))
    with open(Config.TRAIN_DATA_PATH, "w") as f:
        f.write("\n".join(all_samples[:split_idx]))
    with open(Config.VALID_DATA_PATH, "w") as f:
        f.write("\n".join(all_samples[split_idx:]))

if __name__ == "__main__":
    main()