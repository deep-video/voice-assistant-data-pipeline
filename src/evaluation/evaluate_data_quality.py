import json
import sys
import os
import torch
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.intent_descriptions import INTENT_DESCRIPTIONS

# 加载模型
model_path = "d:\\second_domain\\qwen3-4b"
print(f"正在加载模型：{model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成\n")

# 评估提示词
EVAL_PROMPT = """你是数据质量评估员，需要判断语音助手训练数据是否保留。

**意图**：{intent}
**意图描述**：{intent_description}
**输入**：{user_input}

**评估标准**：
1. 必须包含与意图相关的隐含词（能让人联想到该意图的关键词）
   - 例如音乐推荐意图："听"、"歌"、"音乐"、"曲"、"旋律"、"首"等
   - 仅有动作词（如"推荐"、"播放"）不足以判断，必须包含领域相关词
2. 表达自然，像真人对语音助手说话
3. 长度 5-20 字

**输出格式**（只输出 JSON）：
{{"keep": true/false, "reason": "简短理由"}}
"""


def evaluate_item(item):
    """评估单条数据"""
    user_input = item["user_input"]
    intent = item["intent"]
    intent_description = INTENT_DESCRIPTIONS.get(intent, "无描述")
    
    prompt = EVAL_PROMPT.format(
        intent=intent,
        intent_description=intent_description,
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
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0
        )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    # 解析结果
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            result = json.loads(content[start:end])
            keep = result.get("keep", False)
            reason = result.get("reason", "")
        else:
            keep = False
            reason = "解析失败"
    except:
        keep = False
        reason = "解析失败"
    
    item["eval_keep"] = keep
    item["eval_reason"] = reason
    
    return keep, reason


def evaluate_data(data: list):
    """
    评估数据质量（本地模型版本）
    
    参数：
    - data: 待评估的数据列表
    
    返回：
    - (keep_data, discard_data): 保留和丢弃的数据列表
    """
    keep_data = []
    discard_data = []
    
    print(f"开始评估 {len(data)} 条数据...\n")
    
    for i, item in enumerate(tqdm(data, desc="评估进度"), 1):
        keep, reason = evaluate_item(item)
        
        if keep:
            keep_data.append(item)
            print(f"[{i}/{len(data)}] ✅ 保留：{item['user_input']} | {reason}")
        else:
            discard_data.append(item)
            print(f"[{i}/{len(data)}] ❌ 丢弃：{item['user_input']} | {reason}")
    
    print(f"\n评估完成！")
    print(f"保留：{len(keep_data)} 条")
    print(f"丢弃：{len(discard_data)} 条")
    print(f"保留率：{len(keep_data)/len(data)*100:.1f}%")
    
    return keep_data, discard_data


if __name__ == "__main__":
    # 示例用法：直接运行此脚本测试
    test_data = [
        {"user_input": "推荐几首好听的歌", "intent": "music_recommendation"},
        {"user_input": "播放音乐", "intent": "music_recommendation"},
    ]
    
    keep, discard = evaluate_data(test_data)
    
    # 保存测试结果
    with open("d:\\second_domain\\src\\core\\quality_evaluation\\test_keep.json", "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    
    with open("d:\\second_domain\\src\\core\\quality_evaluation\\test_discard.json", "w", encoding="utf-8") as f:
        json.dump(discard, f, ensure_ascii=False, indent=2)
