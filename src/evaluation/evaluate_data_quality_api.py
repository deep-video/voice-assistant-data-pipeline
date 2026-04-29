"""
数据质量评估模块 - API版本（支持并发）
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 初始化客户端
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_URL
)

# 评估提示词
EVAL_PROMPT = """你是数据质量评估员，需要为语音助手训练数据打分（0-5分）。

**评估标准**：
1. 意图可推断：能大致推断是{intent}意图（允许口语化导致的模糊）
2. 真实自然：像真人对语音助手说话，口语化
3. 长度合适：5-20字
4. **允许**：省略句、碎片句、结巴、语气词等口语噪声

**意图**：{intent}
**输入**：{user_input}

**打分标准**：
- 5分：意图明确，表达自然，完全符合
- 4分：意图可推断，表达较自然，基本符合
- 3分：意图可大致推断，有一定口语化特征
- 2分：意图模糊，表达不自然
- 1分：意图不明确，难以判断
- 0分：完全不符合要求

**输出格式**（只输出JSON）：
{{"score": 0-5, "reason": "简短理由"}}
"""

def evaluate_item(item, idx):
    """评估单条数据"""
    user_input = item["user_input"]
    intent = item["intent"]
    
    prompt = EVAL_PROMPT.format(intent=intent, user_input=user_input)
    
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
            score = result.get("score", 0)
            reason = result.get("reason", "")
        else:
            score = 0
            reason = "解析失败"
    except Exception as e:
        score = 0
        reason = f"错误: {str(e)}"
    
    item["eval_score"] = score
    item["eval_reason"] = reason
    
    return idx, item, score, reason

def evaluate_data(data: list, max_workers: int = 8, score_threshold: int = 3):
    """
    评估数据质量
    
    参数：
    - data: 待评估的数据列表
    - max_workers: 并发线程数
    - score_threshold: 保留分数阈值（>=该分数保留）
    
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
            idx, item, score, reason = future.result()
            results[idx] = item
            
            if score >= score_threshold:
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