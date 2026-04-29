"""
意图分类评估：计算每个意图的 precision, recall, F1, FPR
使用训练集/测试集划分（8:2），考虑混淆意图边界样本
"""

import json
import torch
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import random

# 设置随机种子以保证可复现性
random.seed(42)
np.random.seed(42)

# 加载模型
model_path = "d:\\second_domain\\qwen3-1.7b"
print(f"加载模型：{model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 意图列表（添加 others）
ALL_INTENTS = [
    "music_recommendation",
    "car_control",
    "phone_call",
    "navigation",
    "travel_planning",
    "history_query",
    "weather_query",
    "traffic_query",
    "emergency_assistance",
    "poi_recommendation",
    "transport_query",
    "media_recommendation",
    "others"
]

# 意图描述
INTENT_DESCRIPTIONS = {
    "music_recommendation": "用户想要系统推荐/建议音乐，包含询问听什么、推荐几首、旋律、歌、音乐等表达",
    "car_control": "用户想要控制车辆功能，如空调、车窗、座椅、灯光等",
    "phone_call": "用户想要拨打电话",
    "navigation": "用户想要设置导航或查询路线",
    "travel_planning": "用户想要系统规划旅行路线或行程",
    "history_query": "用户想要查询历史记录，如播放历史、搜索历史",
    "weather_query": "用户想要查询天气信息",
    "traffic_query": "用户想要查询交通信息，如路况、公交、地铁",
    "emergency_assistance": "用户想要寻求紧急帮助",
    "poi_recommendation": "用户想要系统推荐/建议兴趣点（POI）",
    "transport_query": "用户想要查询交通方式信息，如飞机、火车、公交",
    "media_recommendation": "用户想要系统推荐/建议媒体内容（视频、播客，不包括音乐）",
    "others": "不属于以上任何意图的其他输入"
}

# 加载数据
data_file = "d:\\second_domain\\src\\core\\final_dataset\\all_intents_dataset.json"
print(f"加载数据：{data_file}")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"数据总量：{len(data)} 条\n")

# 先划分数据集
print("="*70)
print("数据集划分")
print("="*70)

# 按意图分组
intent_groups = {}
boundary_samples = []

for item in data:
    intent = item["intent"]
    
    # 检查是否是边界样本
    is_boundary = item.get("boundary_with", None) is not None
    
    if is_boundary:
        if intent not in intent_groups:
            intent_groups[intent] = []
        boundary_samples.append(item)
    else:
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(item)

print(f"\n普通样本分布:")
for intent, items in sorted(intent_groups.items()):
    print(f"  {intent}: {len(items)} 条")

print(f"\n边界样本：{len(boundary_samples)} 条")

# 对每个意图的普通样本进行 8:2 划分
train_data = []
test_data = []

for intent, items in intent_groups.items():
    if len(items) < 2:
        train_data.extend(items)
        continue
    
    train_items, test_items = train_test_split(
        items,
        test_size=0.2,
        random_state=42
    )
    
    train_data.extend(train_items)
    test_data.extend(test_items)

# 边界样本也按 8:2 划分
if boundary_samples:
    boundary_groups = {}
    for item in boundary_samples:
        boundary_with = item["boundary_with"]
        if boundary_with not in boundary_groups:
            boundary_groups[boundary_with] = []
        boundary_groups[boundary_with].append(item)
    
    print(f"\n边界样本按混淆意图分布:")
    for boundary_intent, items in sorted(boundary_groups.items()):
        print(f"  {boundary_intent}: {len(items)} 条")
        
        if len(items) < 2:
            train_data.extend(items)
        else:
            train_items, test_items = train_test_split(
                items,
                test_size=0.2,
                random_state=42
            )
            train_data.extend(train_items)
            test_data.extend(test_items)

print(f"\n数据集划分结果:")
print(f"  训练集：{len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
print(f"  测试集：{len(test_data)} 条 ({len(test_data)/len(data)*100:.1f}%)")

# 保存划分结果
train_file = "d:\\second_domain\\src\\core\\final_dataset\\train_set.json"
test_file = "d:\\second_domain\\src\\core\\final_dataset\\test_set.json"

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"\n训练集保存到：{train_file}")
print(f"测试集保存到：{test_file}")

# 仅对测试集进行评估
print("\n" + "="*70)
print("仅对测试集进行评估")
print("="*70)

# 评估提示词
EVAL_PROMPT = """你是意图分类器，需要判断用户输入的意图类别。

可选意图及其描述：
{intent_descriptions}

用户输入：{user_input}

请判断该输入属于哪个意图，只输出意图名称（不要输出其他内容）。

意图名称："""

# 构建意图描述字符串
intent_descriptions_str = "\n".join([
    f"- {intent}: {desc}"
    for intent, desc in INTENT_DESCRIPTIONS.items()
])

# 存储预测结果
pred_results = []

print(f"\n开始评估测试集 {len(test_data)} 条数据...\n")

for i, item in enumerate(tqdm(test_data, desc="评估进度"), 1):
    user_input = item["user_input"]
    true_intent = item["intent"]
    
    prompt = EVAL_PROMPT.format(
        intent_descriptions=intent_descriptions_str,
        user_input=user_input
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0
        )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n").strip()
    
    # 提取预测的意图
    pred_intent = content.split("\n")[0].strip()
    
    # 如果预测结果不在意图列表中，尝试匹配
    if pred_intent not in ALL_INTENTS:
        matched = False
        for intent in ALL_INTENTS:
            if intent in pred_intent.lower() or intent.replace("_", " ") in pred_intent.lower():
                pred_intent = intent
                matched = True
                break
        
        if not matched:
            pred_intent = "others"
    
    pred_results.append({
        "user_input": user_input,
        "true_intent": true_intent,
        "pred_intent": pred_intent,
        "correct": true_intent == pred_intent,
        "is_boundary": item.get("boundary_with", None) is not None,
        "boundary_with": item.get("boundary_with", None)
    })
    
    # 每 100 条打印一次进度
    if i % 100 == 0:
        correct = sum(1 for r in pred_results if r["correct"])
        print(f"\n[{i}/{len(test_data)}] 当前准确率：{correct/i*100:.1f}%")

# 保存详细预测结果
output_file = "d:\\second_domain\\src\\core\\final_dataset\\test_predictions.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(pred_results, f, ensure_ascii=False, indent=2)
print(f"\n测试集预测结果保存到：{output_file}")

# 使用测试集计算指标
print("\n" + "="*70)
print("测试集评估指标")
print("="*70)

true_labels = [item["true_intent"] for item in pred_results]
pred_labels = [item["pred_intent"] for item in pred_results]

# 计算每个意图的指标
intent_metrics = {}

for intent in ALL_INTENTS:
    # 二分类：该意图 vs 非该意图
    y_true_binary = [1 if label == intent else 0 for label in true_labels]
    y_pred_binary = [1 if label == intent else 0 for label in pred_labels]
    
    # 计算指标
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # 计算 FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    intent_metrics[intent] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "support": sum(y_true_binary)  # 该意图的真实样本数
    }
    
    print(f"\n{intent}:")
    print(f"  样本数：{intent_metrics[intent]['support']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")

# 计算宏平均指标
macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

# 计算加权平均指标
weighted_precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
weighted_recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

# 总体准确率
total_accuracy = sum(1 for item in pred_results if item["correct"]) / len(pred_results)

print("\n" + "="*70)
print("总体指标（测试集）")
print("="*70)
print(f"测试集样本数：{len(pred_results)}")
print(f"正确预测：{sum(1 for item in pred_results if item['correct'])}")
print(f"总体准确率：{total_accuracy:.4f}")
print(f"宏平均 Precision: {macro_precision:.4f}")
print(f"宏平均 Recall:    {macro_recall:.4f}")
print(f"宏平均 F1:        {macro_f1:.4f}")
print(f"加权 Precision:   {weighted_precision:.4f}")
print(f"加权 Recall:      {weighted_recall:.4f}")
print(f"加权 F1:          {weighted_f1:.4f}")

# 保存指标到 txt 文件
metrics_file = "d:\\second_domain\\src\\core\\final_dataset\\test_metrics.txt"
with open(metrics_file, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write("意图分类评估报告（测试集）\n")
    f.write("="*70 + "\n\n")
    f.write(f"测试集样本数：{len(pred_results)}\n")
    f.write(f"正确预测：{sum(1 for item in pred_results if item['correct'])}\n")
    f.write(f"总体准确率：{total_accuracy:.4f}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("各意图详细指标\n")
    f.write("-"*70 + "\n\n")
    
    for intent in ALL_INTENTS:
        metrics = intent_metrics[intent]
        f.write(f"{intent}:\n")
        f.write(f"  样本数 (Support): {metrics['support']}\n")
        f.write(f"  Precision:        {metrics['precision']:.4f}\n")
        f.write(f"  Recall:           {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:         {metrics['f1']:.4f}\n")
        f.write(f"  FPR:              {metrics['fpr']:.4f}\n\n")
    
    f.write("-"*70 + "\n")
    f.write("总体指标\n")
    f.write("-"*70 + "\n\n")
    f.write(f"宏平均 Precision: {macro_precision:.4f}\n")
    f.write(f"宏平均 Recall:    {macro_recall:.4f}\n")
    f.write(f"宏平均 F1:        {macro_f1:.4f}\n")
    f.write(f"加权 Precision:   {weighted_precision:.4f}\n")
    f.write(f"加权 Recall:      {weighted_recall:.4f}\n")
    f.write(f"加权 F1:          {weighted_f1:.4f}\n")
    f.write(f"总体准确率：      {total_accuracy:.4f}\n")

print(f"\n指标保存到：{metrics_file}")
