# 全局参数配置
class Config:
    # 数据采样
    SAMPLE_SIZE = 100000  # 正负样本各采样数量
    TEST_SIZE = 5000      # 最终分类任务需要预测的fineweb数据量
    
    TRAIN_DATA_PATH = "train.txt"
    VALID_DATA_PATH = "valid.txt"
    MODEL_PATH = "model.bin"
    PREDICT_OUTPUT_PATH = "result.txt"
    
    # FastText超参数
    EPOCHS = 10
    LEARNING_RATE = 0.1
    WORD_NGRAMS = 2
    DIM = 100
    LOSS = "hs"