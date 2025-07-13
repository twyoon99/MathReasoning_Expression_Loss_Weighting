import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from base_dataset import BaseMathDataset
from torch.optim import AdamW

# Baseline Trainer (수학 토큰 가중치 없음)
class BaselineTrainer(Trainer):
    def __init__(self, *args, weight_decay=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "bias" not in n and "LayerNorm.weight" not in n],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if "bias" in n or "LayerNorm.weight" in n],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

# 모델 및 학습 설정 
model_name = "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)


train_dataset = BaseMathDataset("/home/woong/MediTOD_DST/math-split-slm/data/train_socratic.jsonl", tokenizer)


training_args = TrainingArguments(
    output_dir="/home/woong/MediTOD_DST/math-split-slm/base_outputs/checkpoints/baseline_model_qwen2_exp1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_steps=200,
    num_train_epochs=5,
    logging_steps=100,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=True,
    load_best_model_at_end=False
)

trainer = BaselineTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    weight_decay=0.01
)

trainer.train()
model.save_pretrained("/home/woong/MediTOD_DST/math-split-slm/base_outputs/checkpoints/baseline_model_qwen2_exp1")