import fasttext
from datasets import load_dataset
import warnings
import re

# 配置参数
CONFIG = {
    "model_path": "model.bin",
    "output_file": "labeled_fineweb.txt",
    "sample_size": 5000,
    "max_text_length": 1000  # 防止内存溢出
}

def clean_text(text):
    """文本清洗函数"""
    text = re.sub(r'\s+', ' ', text)  # 合并多余空白字符
    return text.strip()[:CONFIG["max_text_length"]]

def load_model():
    """安全加载模型函数"""   
    # 方法2：或者使用新API（如果可用）
    return fasttext.FastText.load_model(CONFIG["model_path"])

def process_data():
    """主处理函数"""
    print("正在加载模型...")
    model = load_model()
    
    print("正在获取fineweb数据...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True
    )
    
    print("正在处理并预测标签...")
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            if i >= CONFIG["sample_size"]:
                break
                
            text = clean_text(item["text"])
            label = model.predict(text)[0][0] 
            
            f.write(f"{label}\t{text}\n")

            if (i+1) % 500 == 0:
                print(f"已处理 {i+1}/{CONFIG['sample_size']} 条")

    print(f"\n完成!结果已保存至 {CONFIG['output_file']}")

if __name__ == "__main__":
    # 显示友好提示
    print("="*50)
    print("FastText 数据标注工具")
    print(f"目标: 从fineweb生成 {CONFIG['sample_size']} 条带标签数据")
    print("="*50)
    
    process_data()