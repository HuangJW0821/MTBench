import json
from tqdm import tqdm
from transformers import MambaConfig, MambaModel, AutoTokenizer
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import torch.nn as nn
from model_setting import MambaForRegression

# 数据预处理函数
def preprocess_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = examples["toxicity"]  # 保留标签
    return tokens


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = predictions.flatten()  # 展平预测结果
    # # 应用 Sigmoid 将输出值限制在 [0, 1]
    # predictions = 1 / (1 + np.exp(-predictions))

    # 计算均方误差
    mse = ((predictions - labels) ** 2).mean()
    return {"mse": mse}

output_dir = "./results"

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
config = MambaConfig(
    vocab_size=len(tokenizer.vocab),  # 假设 tokenizer.vocab 大小为 30,000
    n_positions=128,                 # 假设输入文本较短
    n_embd=256,                      # 降低嵌入维度
    n_layer=6,                       # 减少层数
    n_head=4,                        # 保持头数为 4
    n_inner=1024,                    # 减少前馈层中间大小
)
model = MambaForRegression(config).cuda()

dataset = load_dataset("TheMrguiller/jigsaw-unintended-bias-in-toxicity-classification")["test"]
dataset = dataset.select(range(1000))

# 对数据集进行预处理
processed_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = processed_dataset
# train_size = int(0.8 * len(processed_dataset))
# train_dataset = processed_dataset.select(range(train_size))
# eval_dataset = processed_dataset.select(range(train_size, len(processed_dataset)))

training_args = TrainingArguments(
    output_dir=output_dir,
    # evaluation_strategy="epoch",       # 每个 epoch 评估
    evaluation_strategy='no',
    learning_rate=2e-5,                # 学习率
    per_device_train_batch_size=8,     # 每个设备训练批量大小
    per_device_eval_batch_size=16,     # 每个设备验证批量大小
    num_train_epochs=5,                # 训练轮数
    weight_decay=0.01,                 # 权重衰减
    logging_dir="./logs",              # 日志目录
    save_steps=10_000,                 # 保存步数
    save_total_limit=2,                # 保存最大模型数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,   # 训练数据集
    # eval_dataset=eval_dataset,    # 验证数据集
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir+"/final_model")
model.config.save_pretrained(output_dir+"/final_model")


texts = ["Fuck you Fuck you", "Good morning and good night"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
outputs = model(**inputs)
predictions = torch.sigmoid(outputs).squeeze().tolist()
print(predictions)  # 输出：[0.89, 0.12] 表示第一个是高 toxicity，第二个是低 toxicity