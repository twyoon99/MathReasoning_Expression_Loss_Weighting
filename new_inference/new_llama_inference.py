import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm 

model_name = "meta-llama/Meta-Llama-3-8B"
lora_path = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/checkpoints/new_model_llama_exp2/checkpoint-2335"
data_path = "/home/woong/MediTOD_DST/math-split-slm/data/test_socratic.jsonl"
output_path = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/predictions_new_llama_exp2_2335.jsonl"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

def generate_answer(question):
    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()

# Run inference with tqdm progress bar
predictions = []
with open(data_path, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Inference Progress"):
    ex = json.loads(line)
    question = ex["question"]
    gold_answer = ex["answer"]
    pred_answer = generate_answer(question)
    predictions.append({
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer
    })

with open(output_path, 'w') as f:
    for item in predictions:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")