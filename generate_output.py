import json
from tqdm import tqdm
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch
import random

# Load model
model_name = "mamba-chat"
tokenizer_path = "havenhq/mamba-chat"
model_path = "havenhq/mamba-chat"
output_path = f'./eval_data/{model_name}.jsonl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
# model = MambaForCausalLM.from_pretrained(model_path)
model = MambaLMHeadModel.from_pretrained("havenhq/mamba-chat", device="cuda", dtype=torch.float16)
model.to(device)

# Start generate outputs
print(f"[*] Start generating outputs from {model_name} ...")
data_path = "./DataBench.jsonl"

with open(data_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# lines = random.sample(lines, 1000)
results = []
batch_size = 64

for i in tqdm(range(0, len(lines), batch_size), total=len(lines)//batch_size, desc="Evaluating", unit="batch"):
    batch_lines = lines[i:i+batch_size]
    input_texts = [line['text'] for line in batch_lines]
    
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    input_ids = input_ids.to(device)

    # outputs = model.generate(input_ids, max_new_tokens=50)
    outputs = model.generate(input_ids, max_length=200)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)


    for idx, line in enumerate(batch_lines):
        result = {
            "index": line["index"],
            "text": line["text"],
            "from": line["from"],
            "output": decoded_outputs[idx]
        }
        results.append(result)

    # Print some outputs for debug
    if i==0:
        print(input_texts, decoded_outputs)

# Save outputs
print("[*] Saving outputs ...")
with open(output_path, "w", encoding="utf-8") as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")
print(f"[*] Finished generate! Output save at {output_path}.")
