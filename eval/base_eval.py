import json
import re

PRED_PATH = "/home/woong/MediTOD_DST/math-split-slm/base_outputs/predictions_base_qwen_exp1.jsonl"
RESULT_PATH = "/home/woong/MediTOD_DST/math-split-slm/base_outputs/base_results_qwen_exp1.json"

def extract_final_number(text):
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def extract_all_numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))

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

# 평가 지표 계산
results = []
total, correct = 0, 0
f1_scores = []

with open(PRED_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        gold_ans = data["gold_answer"]
        pred_ans = data["pred_answer"]

        # Accuracy 계산용
        gold = extract_final_number(gold_ans)
        pred = extract_final_number(pred_ans)
        is_correct = gold == pred if gold and pred else False
        if is_correct:
            correct += 1

        # Numeric F1 계산용
        gold_nums = extract_all_numbers(gold_ans)
        pred_nums = extract_all_numbers(pred_ans)
        f1 = compute_numeric_f1(gold_nums, pred_nums)
        f1_scores.append(f1)

        total += 1
        results.append({
            "question": data["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "numeric_f1": round(f1, 4)
        })

accuracy = correct / total * 100 if total > 0 else 0
avg_f1 = sum(f1_scores) / total if total > 0 else 0

# 콘솔 출력
print(f"📌 Accuracy      : {correct}/{total} = {accuracy:.2f}%")
print(f"📌 Numeric F1 Avg: {avg_f1:.4f}")

# JSON 저장
with open(RESULT_PATH, 'w') as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_numeric_f1": avg_f1,
        "results": results
    }, f, ensure_ascii=False, indent=2)

# import json
# import re

# PRED_PATH = "/home/woong/MediTOD_DST/math-split-slm/base_outputs/predictions_base_exp6_2335.jsonl"
# RESULT_PATH = "/home/woong/MediTOD_DST/math-split-slm/base_outputs/base_result_exp6_2335.json"

# # 숫자 추출 함수 (소수 포함)
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

# print(f"Base Accuracy: {correct}/{total} = {accuracy:.2f}%")

# # JSON 결과 저장
# with open(RESULT_PATH, 'w') as f:
#     json.dump({
#         "accuracy": accuracy,
#         "correct": correct,
#         "total": total,
#         "results": results
#     }, f, ensure_ascii=False, indent=2)