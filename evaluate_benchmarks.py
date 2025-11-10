import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 根目录，确保是在 personality/ 下运行 ===
base_path = "./results/benchmark"

# === 模型文件夹名称 ===
model_folders = ["N性格模型", "S性格模型", "原始基座模型"]

# === 数据集文件名映射（新增 GSM8K） ===
files = {
    "ARC (easy)": "arc_easy_test800_results.csv",
    "BoolQ": "boolq_train800_results.csv",
    "GSM8K": "gsm8k_test800_results.csv"
}

# === 提取函数 ===
def extract_upper_letter(text):
    match = re.search(r'\b([A-D])\b', str(text).upper())
    return match.group(1) if match else None

def extract_bool(text):
    if isinstance(text, str):
        text_lower = text.lower()
        if 'true' in text_lower:
            return True
        elif 'false' in text_lower:
            return False
    elif isinstance(text, bool):
        return text
    return None

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

# === GSM8K：更宽松的数值命中评测（与 evaluate一个 保持一致） ===
def extract_numbers(text):
    """提取数字列表：移除逗号和美元符号，匹配整数/小数（与 evaluate一个 保持一致）"""
    text = str(text).replace(",", "").replace("$", "")
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]

def gsm8k_accuracy_from_numbers(df):
    """只要标签中的第一个数字出现在预测数字列表中就算正确；只统计能解析出数字的样本"""
    correct, total = 0, 0
    for _, row in df.iterrows():
        label_nums = extract_numbers(row["label"])
        pred_nums = extract_numbers(row["prediction"])
        if not label_nums or not pred_nums:
            continue
        label = label_nums[0]
        if label in pred_nums:
            correct += 1
        total += 1
    acc = correct / total if total else 0.0
    return round(acc, 4), correct, total

# === 收集所有结果 ===
all_results = []

for model_name in model_folders:
    model_path = os.path.join(base_path, model_name)

    for dataset_name, filename in files.items():
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        if dataset_name == "ARC (easy)":
            df["label_clean"] = df["label"].apply(extract_upper_letter)
            df["prediction_clean"] = df["prediction"].apply(extract_upper_letter)
            df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
            metrics = compute_metrics(df_valid["label_clean"], df_valid["prediction_clean"])

        elif dataset_name == "BoolQ":
            df["label_clean"] = df["label"].apply(extract_bool)
            df["prediction_clean"] = df["prediction"].apply(extract_bool)
            df_valid = df.dropna(subset=["label_clean", "prediction_clean"])
            metrics = compute_metrics(df_valid["label_clean"], df_valid["prediction_clean"])

        elif dataset_name == "GSM8K":
            # 使用更宽松的数值命中评测
            accuracy, correct, total = gsm8k_accuracy_from_numbers(df)
            metrics = {
                "accuracy": accuracy,
                "precision": None,
                "recall": None,
                "f1": None,
                # 可选：也把可解析样本统计到表里，便于审计（不想展示可以去掉这两列）
                #"parsed_correct": correct,
                #"parsed_total": total,
            }

        all_results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            **metrics
        })

# === 输出为 DataFrame 结果表 ===
df_metrics = pd.DataFrame(all_results)
print(df_metrics)

# === 保存结果到 txt 文件 ===
output_path = os.path.join(base_path, "benchmark_metrics_summary.txt")

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_metrics.iterrows():
        f.write(
            f"\n📌 Model: {row['Model']}\n"
            f"📊 Dataset: {row['Dataset']}\n"
            f"✅ Accuracy: {row['accuracy']}\n"
            f"✅ Precision: {row['precision']}\n"
            f"✅ Recall: {row['recall']}\n"
            f"✅ F1 Score: {row['f1']}\n"
            f"{'-'*40}\n"
        )

print(f"\n📁 已将结果保存到：{output_path}")
