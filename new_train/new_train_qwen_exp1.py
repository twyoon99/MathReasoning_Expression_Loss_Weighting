import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from base_dataset import BaseMathDataset
from torch.optim import AdamW

# 수학 표현을 인덱스 기반 추출
def extract_math_expressions(text):
    matches = []
    latex_patterns = [
        (r'\$([^\n]+?)\$', 'inline'),
        (r'\\\[(.+?)\\\]', 'display_brackets'),
        (r'\$\$(.+?)\$\$', 'display_dollars'),
        (r'\\begin\{(align\*?|equation\*?|gather\*?)\}(.+?)\\end\{\1\}', 'env')
    ]
    for pattern, label in latex_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1) if label != 'env' else match.group(2)
            start, end = match.start(), match.end()
            if label == 'inline':
                alphabetic_count = sum(c.isalpha() for c in content)
                if alphabetic_count / len(content) > 0.5:
                    continue
            matches.append((start, end, content))

    text_math_pattern = r'(?<![A-Za-z0-9])(?:(?:\d+(?:\.\d+)?)|[A-Za-z])\b(?:[^\S\r\n]*(?:[=+\-*/\u00F7\u00D7\u00B7^%<>!]+)[^\S\r\n]*\b(?:(?:\d+(?:\.\d+)?)|[A-Za-z])\b)+(?![A-Za-z0-9])'
    for match in re.finditer(text_math_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))

    boxed_pattern = r'(\\boxed\{.*?\})'
    for match in re.finditer(boxed_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))

    number_pattern = r'\b\d+(?:\.\d+)?\b'
    for match in re.finditer(number_pattern, text):
        matches.append((match.start(), match.end(), match.group(0)))

    matches = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    final_matches = []
    last_end = -1
    for start, end, content in matches:
        if start >= last_end:
            final_matches.append((start, end, content))
            last_end = end
    return final_matches

# 각 토큰이 수학 표현인지 아닌지 마스크 생성 
def mark_math_tokens(text, tokenizer, input_ids_sample):
    math_spans = extract_math_expressions(text)
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids_sample]
    offsets = []
    pointer = 0
    for tok in decoded_tokens:
        tok = tok.strip()
        if not tok:
            offsets.append((pointer, pointer))
            continue
        start = text.find(tok, pointer)
        if start == -1:
            start = pointer
        end = start + len(tok)
        offsets.append((start, end))
        pointer = end

    math_mask = [False] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        for m_start, m_end, _ in math_spans:
            if start < m_end and end > m_start:
                math_mask[i] = True
                break
    return math_mask

# 수학 표현에 더 큰 가중치를 부여하여 loss 계산 
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, math_weight=1.5, weight_decay=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.math_weight = math_weight
        self.weight_decay = weight_decay

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        all_weights = []
        batch_size, seq_len = shift_input_ids.size()

        for i in range(batch_size):
            input_ids_sample = shift_input_ids[i]
            full_input_ids = input_ids[i]
            text = tokenizer.decode(full_input_ids, skip_special_tokens=False)

            math_mask = mark_math_tokens(text, tokenizer, input_ids_sample)

            # padding or trimming
            if len(math_mask) < input_ids_sample.size(0):
                math_mask += [False] * (input_ids_sample.size(0) - len(math_mask))
            elif len(math_mask) > input_ids_sample.size(0):
                math_mask = math_mask[:input_ids_sample.size(0)]

            weights = [self.math_weight if is_math else 1.0 for is_math in math_mask]
            all_weights.extend(weights)

        weight_tensor = torch.tensor(all_weights, device=loss.device)
        assert weight_tensor.size() == loss.size(), f"Weight/loss size mismatch: {weight_tensor.size()} vs {loss.size()}"

        weighted_loss = (loss * weight_tensor).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

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
    output_dir="/home/woong/MediTOD_DST/math-split-slm/new_outputs/checkpoints/new_model_qwen_exp1",
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

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    math_weight=1.5,
    weight_decay=0.01
)

trainer.train()
model.save_pretrained("/home/woong/MediTOD_DST/math-split-slm/new_outputs/checkpoints/new_model_qwen_exp1")