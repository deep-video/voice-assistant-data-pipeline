"""
数据质量评估模块 - API 版本（支持并发）
"""

import json
import sys
import os
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME
from src.config.intent_descriptions import INTENT_DESCRIPTIONS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 初始化客户端
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_URL
)

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

def evaluate_item(item, idx):
    """评估单条数据"""
    user_input = item["user_input"]
    intent = item["intent"]
    intent_description = INTENT_DESCRIPTIONS.get(intent, "无描述")
    
    prompt = EVAL_PROMPT.format(
        intent=intent,
        intent_description=intent_description,
        user_input=user_input
    )
    
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=128
        )
        
        content = response.choices[0].message.content
        
        # 解析结果
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            result = json.loads(content[start:end])
            keep = result.get("keep", False)
            reason = result.get("reason", "")
        else:
            keep = False
            reason = "解析失败"
    except Exception as e:
        keep = False
        reason = f"错误：{str(e)}"
    
    item["eval_keep"] = keep
    item["eval_reason"] = reason
    
    return idx, item, keep, reason

def evaluate_data(data: list, max_workers: int = 8):
    """
    评估数据质量
    
    参数：
    - data: 待评估的数据列表
    - max_workers: 并发线程数
    
    返回：
    - (keep_data, discard_data): 保留和丢弃的数据列表
    """
    keep_data = []
    discard_data = []
    results = [None] * len(data)
    
    print(f"开始评估 {len(data)} 条数据...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(data):
            futures.append(executor.submit(evaluate_item, item, idx))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
            idx, item, keep, reason = future.result()
            results[idx] = item
            
            if keep:
                keep_data.append(item)
            else:
                discard_data.append(item)
    
    return keep_data, discard_data


if __name__ == "__main__":
    # 加载数据
    with open("d:\\second_domain\\llm_seed_pipeline\\expanded_data\\music_recommendation_expanded.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 并发评估
    keep_data, discard_data = evaluate_data(data)
    
    # 保存结果
    with open("d:\\second_domain\\llm_seed_pipeline\\quality_evaluation\\keep.json", "w", encoding="utf-8") as f:
        json.dump(keep_data, f, ensure_ascii=False, indent=2)
    
    with open("d:\\second_domain\\llm_seed_pipeline\\quality_evaluation\\discard.json", "w", encoding="utf-8") as f:
        json.dump(discard_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成！")
    print(f"保留：{len(keep_data)} 条")
    print(f"丢弃：{len(discard_data)} 条")
    print(f"保留率：{len(keep_data)/len(data)*100:.1f}%")
