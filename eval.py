import json
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
import torch
from model_setting import MambaForRegression
from safetensors.torch import load_file

# Replace model_name as you need
model_name = "mamba-130m-hf"
data_path = f"./eval_data/{model_name}.jsonl"
output_path = f"./eval_result/{model_name}.jsonl"

# Load MTBench
print("[*] Initialzing MTBench model ...")
MTBench_path = "./results/final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MTBench_path)
config = AutoConfig.from_pretrained(MTBench_path)

model = MambaForRegression(config)
model.load_state_dict(load_file(MTBench_path+"/model.safetensors"))
model.to(device)
model.eval()
print("[*] MTBench model initialized.")

# MTBench predict function
def predict_toxicity(model, inputs_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    return outputs.squeeze().cpu().numpy()

# Start evaluation
print(f"[*] Start evaluating from {data_path}")

with open(data_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# lines = random.sample(lines, 1000)
results = []
batch_size = 64
final_result = 0

for i in tqdm(range(0, len(lines), batch_size), total=len(lines)//batch_size, desc="Evaluating", unit="batch"):
    batch_lines = lines[i:i+batch_size]
    input_texts = [line['output'] for line in batch_lines]
    
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    input_ids = input_ids.to(device)

    outputs = predict_toxicity(model, input_ids)

    for idx, line in enumerate(batch_lines):
        result = {
            "index": line["index"],
            "text": line["text"],
            "from": line["from"],
            "output": line["output"],
            "toxicity": float(outputs[idx])
        }
        results.append(result)
        final_result += outputs[idx]
    return

print("[*] Final Result: " + str(final_result/len(lines)))
print(final_result/len(lines))

while(1):
    input_texts = input("human: ")
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    input_ids = input_ids.to(device)

    outputs = predict_toxicity(model, input_ids)
    print("assistant: " + str(outputs))

# Save outputs
# print("[*] Saving outputs ...")
# with open(output_path, "w", encoding="utf-8") as f:
#     for result in results:
#         json.dump(result, f, ensure_ascii=False)
#         f.write("\n")
# print(f"[*] Finished generate! Output save at {output_path}.")

