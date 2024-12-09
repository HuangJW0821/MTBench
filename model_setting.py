from transformers import MambaConfig, MambaModel, AutoTokenizer
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import torch.nn as nn

class MambaForRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mamba = MambaModel(config)
        # 延迟初始化 Linear 层
        self.regression_head = None

    def forward(self, input_ids, attention_mask=None):
        # 获取隐藏状态
        outputs = self.mamba(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 动态初始化 regression_head
        if self.regression_head is None:
            self.regression_head = nn.Linear(hidden_states.size(-1), 1).to(hidden_states.device)
        
        # 回归输出
        regression_output = self.regression_head(hidden_states[:, -1, :])
        return regression_output