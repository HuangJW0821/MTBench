from transformers import AutoConfig, AutoTokenizer
import torch
from model_setting import MambaForRegression
from safetensors.torch import load_file

torch.backends.cudnn.deterministic = True

def predict_toxicity(texts, model, tokenizer, device="cuda"):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model.to('cuda')
    inputs = {key: value.to('cuda') for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.squeeze().cpu().numpy()

# 指定保存模型的路径
model_path = "./results/final_model"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

model = MambaForRegression(config)
model.load_state_dict(load_file(model_path+"/model.safetensors"))
model.eval()

texts = ["This movie is fantastic!", "I did not like this film at all."]
results = predict_toxicity(texts, model, tokenizer)
print(results)

while(1):
    user_input = input("Human:")
    print(predict_toxicity(user_input, model, tokenizer))

