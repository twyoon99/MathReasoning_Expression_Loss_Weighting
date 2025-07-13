import json
import re

PRED_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/predictions_new_qwen_exp2.jsonl"
RESULT_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/new_results_qwen_exp2.json"

def extract_final_number(text):
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def extract_all_numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))

def extract_math_spans(text):
    patterns = [
        r"\$.*?\$",                  # inline LaTeX
        r"\\\[.*?\\\]",              # display math
        r"\$\$.*?\$\$",              # double dollar
        r"\\boxed\{.*?\}",           # boxed expressions
        r"\d+(?:\.\d+)?",            # plain numbers
    ]
    spans = []
    for pat in patterns:
        spans.extend(re.findall(pat, text))
    return set(spans)

def compute_numeric_f1(gold_nums, pred_nums):
    if not gold_nums and not pred_nums:
        return 1.0
    if not gold_nums or not pred_nums:
        return 0.0
    overlap = gold_nums & pred_nums
    precision = len(overlap) / len(pred_nums)
    recall = len(overlap) / len(gold_nums)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_math_span_accuracy(gold_spans, pred_spans):
    if not gold_spans:
        return 1.0 if not pred_spans else 0.0
    overlap = gold_spans & pred_spans
    return len(overlap) / len(gold_spans)

results = []
total, correct = 0, 0
f1_scores = []
math_span_scores = []

with open(PRED_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        gold_ans = data["gold_answer"]
        pred_ans = data["pred_answer"]

        # Accuracy
        gold = extract_final_number(gold_ans)
        pred = extract_final_number(pred_ans)
        is_correct = gold == pred if gold and pred else False
        if is_correct:
            correct += 1

        # Numeric F1
        gold_nums = extract_all_numbers(gold_ans)
        pred_nums = extract_all_numbers(pred_ans)
        f1 = compute_numeric_f1(gold_nums, pred_nums)
        f1_scores.append(f1)

        # Math Span Accuracy
        gold_spans = extract_math_spans(gold_ans)
        pred_spans = extract_math_spans(pred_ans)
        math_acc = compute_math_span_accuracy(gold_spans, pred_spans)
        math_span_scores.append(math_acc)

        total += 1
        results.append({
            "question": data["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "numeric_f1": round(f1, 4),
            "math_span_accuracy": round(math_acc, 4)
        })

accuracy = correct / total * 100 if total > 0 else 0
avg_f1 = sum(f1_scores) / total if total > 0 else 0
avg_math_acc = sum(math_span_scores) / total if total > 0 else 0

print(f"Accuracy          : {correct}/{total} = {accuracy:.2f}%")
print(f"Numeric F1 Avg    : {avg_f1:.4f}")
print(f"Math Span Accuracy: {avg_math_acc:.4f}")

with open(RESULT_PATH, 'w') as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_numeric_f1": avg_f1,
        "avg_math_span_accuracy": avg_math_acc,
        "results": results
    }, f, ensure_ascii=False, indent=2)

"""
평가 지표
Accuracy	마지막 숫자(gold vs pred)가 정확히 일치하는 비율
Numeric F1	숫자 토큰들(gold vs pred)의 겹치는 정도에 대한 F1 score
Math Span Accuracy	수학 표현(LaTeX, boxed, 숫자 등)의 일치 정도 평가
"""

# import json
# import re

# PRED_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/predictions_new_exp6_2335.jsonl"
# RESULT_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/new_results_exp6.json"

# def extract_final_number(text):
#     numbers = re.findall(r"\d+(?:\.\d+)?", text)
#     return numbers[-1] if numbers else None

# def extract_all_numbers(text):
#     return set(re.findall(r"\d+(?:\.\d+)?", text))

# def compute_numeric_f1(gold_nums, pred_nums):
#     if not gold_nums and not pred_nums:
#         return 1.0
#     if not gold_nums or not pred_nums:
#         return 0.0
#     overlap = gold_nums & pred_nums
#     precision = len(overlap) / len(pred_nums)
#     recall = len(overlap) / len(gold_nums)
#     if precision + recall == 0:
#         return 0.0
#     return 2 * precision * recall / (precision + recall)

# # 평가 지표 계산
# results = []
# total, correct = 0, 0
# f1_scores = []

# with open(PRED_PATH, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         gold_ans = data["gold_answer"]
#         pred_ans = data["pred_answer"]

#         # Accuracy 계산용
#         gold = extract_final_number(gold_ans)
#         pred = extract_final_number(pred_ans)
#         is_correct = gold == pred if gold and pred else False
#         if is_correct:
#             correct += 1

#         # Numeric F1 계산용
#         gold_nums = extract_all_numbers(gold_ans)
#         pred_nums = extract_all_numbers(pred_ans)
#         f1 = compute_numeric_f1(gold_nums, pred_nums)
#         f1_scores.append(f1)

#         total += 1
#         results.append({
#             "question": data["question"],
#             "gold": gold,
#             "pred": pred,
#             "correct": is_correct,
#             "numeric_f1": round(f1, 4)
#         })

# accuracy = correct / total * 100 if total > 0 else 0
# avg_f1 = sum(f1_scores) / total if total > 0 else 0

# # 콘솔 출력
# print(f"Accuracy      : {correct}/{total} = {accuracy:.2f}%")
# print(f"Numeric F1 Avg: {avg_f1:.4f}")

# # JSON 저장
# with open(RESULT_PATH, 'w') as f:
#     json.dump({
#         "accuracy": accuracy,
#         "correct": correct,
#         "total": total,
#         "avg_numeric_f1": avg_f1,
#         "results": results
#     }, f, ensure_ascii=False, indent=2)

# """
# Numeric F1 Score

# Precision: 예측 숫자 중 정답과 겹치는 비율
# Recall: 정답 숫자 중 예측과 겹치는 비율
# F1 Score: Precision과 Recall의 조화 평균    
# """
# import json
# import re

# PRED_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/predictions_new_exp6_2335.jsonl"
# RESULT_PATH = "/home/woong/MediTOD_DST/math-split-slm/new_outputs/new_results_exp6.json"

# def extract_final_number(text):
#     numbers = re.findall(r"\d+(?:\.\d+)?", text)
#     return numbers[-1] if numbers else None

# results = []
# total, correct = 0, 0

# with open(PRED_PATH, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         gold = extract_final_number(data["gold_answer"])
#         pred = extract_final_number(data["pred_answer"])

#         is_correct = gold == pred if gold and pred else False
#         if is_correct:
#             correct += 1
#         total += 1

#         results.append({
#             "question": data["question"],
#             "gold": gold,
#             "pred": pred,
#             "correct": is_correct
#         })

# accuracy = correct / total * 100 if total > 0 else 0

# print(f"New_Accuracy: {correct}/{total} = {accuracy:.4f}%")

# # JSON 결과 저장
# with open(RESULT_PATH, 'w') as f:
#     json.dump({
#         "accuracy": accuracy,
#         "correct": correct,
#         "total": total,
#         "results": results
#     }, f, ensure_ascii=False, indent=2)