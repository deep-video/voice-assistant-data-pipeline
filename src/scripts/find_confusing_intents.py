import openai
import json
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME
from src.config.intent_descriptions import INTENT_DESCRIPTIONS, ALL_INTENTS

# 初始化客户端
client = openai.OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_URL
)

# 创建输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "confusing_intents")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_confusing_intents(intent: str, all_intents: list) -> list:
    """让模型自动找出所有可能混淆的意图"""
    intent_desc = INTENT_DESCRIPTIONS[intent]
    prompt = f"""你是意图分类专家，现在有一个意图 "{intent}"：
    - 描述：{intent_desc['description']}
    - 动作：{intent_desc['intent_action']}
    - 反例：{intent_desc['intent_negative_example']}
    
    请从列表中找出所有可能与之混淆的意图，并说明理由：
    {all_intents}
    
    输出 JSON 数组，每个元素包含 "intent" 和 "reason"：
    [
      {{"intent": "intent1", "reason": "混淆原因"}},
      ...
    ]
    """
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是意图分类专家，擅长识别意图边界和混淆点。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 低温度，保证结果准确
                max_tokens=1024
            )
            
            result = response.choices[0].message.content
            
            # 清理 markdown
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            # 提取 JSON
            if "[" in result and "]" in result:
                start_idx = result.find("[")
                end_idx = result.rfind("]") + 1
                result = result[start_idx:end_idx]
            
            confusing_intents = json.loads(result)
            
            return confusing_intents
        except Exception as e:
            print(f"尝试 {attempt+1} 失败：{e}")
            continue
    
    return []

def find_all_confusing_intents():
    """找出所有意图的混淆意图"""
    for intent in ALL_INTENTS:
        print(f"\n{'='*70}")
        print(f"找出与 '{intent}' 混淆的意图...")
        print(f"{'='*70}")
        
        # 排除自身
        other_intents = [i for i in ALL_INTENTS if i != intent]
        confusing_intents = find_confusing_intents(intent, other_intents)
        
        if confusing_intents:
            print(f"\n找到 {len(confusing_intents)} 个混淆意图：")
            for item in confusing_intents:
                print(f"- {item['intent']}: {item['reason']}")
                
            # 保存结果
            output_file = os.path.join(OUTPUT_DIR, f"{intent}_confusing.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(confusing_intents, f, ensure_ascii=False, indent=2)
            
            print(f"\n结果保存到：{output_file}")
        else:
            print("未找到混淆意图")

if __name__ == "__main__":
    # 测试：只生成音乐推荐的混淆意图
    intent = "music_recommendation"
    other_intents = [i for i in ALL_INTENTS if i != intent]
    
    print(f"{'='*70}")
    print(f"找出与 '{intent}' 混淆的意图...")
    print(f"{'='*70}")
    
    confusing_intents = find_confusing_intents(intent, other_intents)
    
    if confusing_intents:
        print(f"\n找到 {len(confusing_intents)} 个混淆意图：")
        for item in confusing_intents:
            print(f"- {item['intent']}: {item['reason']}")
            
        # 保存结果
        output_file = os.path.join(OUTPUT_DIR, f"{intent}_confusing.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(confusing_intents, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果保存到：{output_file}")
    else:
        print("未找到混淆意图")