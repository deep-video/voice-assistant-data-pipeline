import openai
import json
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME
from src.config.intent_descriptions import INTENT_DESCRIPTIONS, ALL_INTENTS
from src.config.prompt_templates import BOUNDARY_SAMPLE_GENERATION_PROMPT

# 初始化客户端
client = openai.OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_URL
)

# 输出目录
BOUNDARY_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'boundary_samples')
CONFUSING_INTENTS_DIR = os.path.join(os.path.dirname(__file__), 'confusing_intents')
os.makedirs(BOUNDARY_SAMPLES_DIR, exist_ok=True)
os.makedirs(CONFUSING_INTENTS_DIR, exist_ok=True)


def find_confusing_intents(intent: str) -> list:
    """让模型自动找出所有可能混淆的意图"""
    intent_desc = INTENT_DESCRIPTIONS[intent]
    other_intents = [i for i in ALL_INTENTS if i != intent]
    
    prompt = f"""你是意图分类专家，现在有一个意图 "{intent}"：
    - 描述：{intent_desc['description']}
    - 动作：{intent_desc['intent_action']}
    - 反例：{intent_desc['intent_negative_example']}
    
    请从列表中找出所有可能与之混淆的意图，并说明理由：
    {other_intents}
    
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
                temperature=0.3,
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
            
            # 保存结果
            output_file = os.path.join(CONFUSING_INTENTS_DIR, f"{intent}_confusing.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(confusing_intents, f, ensure_ascii=False, indent=2)
            
            return confusing_intents
        except Exception as e:
            print(f"尝试 {attempt+1} 失败：{e}")
            continue
    
    return []


def load_confusing_intents(intent: str) -> list:
    """加载已有的混淆意图，如果没有则自动生成"""
    output_file = os.path.join(CONFUSING_INTENTS_DIR, f"{intent}_confusing.json")
    
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"未找到混淆意图文件，正在使用大模型生成...")
    return find_confusing_intents(intent)


def generate_boundary_samples(intent: str, confusing_intent: dict, num_samples: int = 10) -> list:
    """生成边界样本"""
    intent_desc = INTENT_DESCRIPTIONS[intent]
    confusing_intent_name = confusing_intent["intent"]
    confusing_intent_desc = INTENT_DESCRIPTIONS[confusing_intent_name]
    
    prompt = BOUNDARY_SAMPLE_GENERATION_PROMPT.format(
        intent=intent,
        intent_description=intent_desc['description'],
        intent_action=intent_desc['intent_action'],
        intent_negative_example=intent_desc['intent_negative_example'],
        confusing_intent=confusing_intent_name,
        confusing_intent_description=confusing_intent_desc['description'],
        confusing_intent_action=confusing_intent_desc['intent_action'],
        confusing_intent_negative_example=confusing_intent_desc['intent_negative_example'],
        confusing_reason=confusing_intent['reason'],
        num_samples=num_samples
    )
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是语言学家，专门研究人类对车载语音助手的真实说话方式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
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
            
            samples = json.loads(result)
            
            return samples
        except Exception as e:
            print(f"尝试 {attempt+1} 失败：{e}")
            continue
    
    return []


def generate_all_boundary_samples(intent: str, num_samples: int = 10) -> list:
    """为指定意图生成所有边界样本"""
    print(f"生成 '{intent}' 的边界样本...")
    
    # 加载或生成混淆意图
    confusing_intents = load_confusing_intents(intent)
    
    if not confusing_intents:
        print(f"未找到与 '{intent}' 混淆的意图")
        return []
    
    print(f"找到 {len(confusing_intents)} 个混淆意图：")
    for item in confusing_intents:
        print(f"  - {item['intent']}: {item['reason'][:50]}...")
    
    all_boundary_samples = []
    
    # 为每个混淆意图生成边界样本
    for confusing_intent in confusing_intents:
        print(f"\n针对 '{confusing_intent['intent']}' 生成边界样本...")
        samples = generate_boundary_samples(intent, confusing_intent, num_samples=num_samples)
        
        if samples:
            # 为每条样本添加意图字段（使用规则添加）
            samples_with_intent = [
                {
                    "user_input": sample,
                    "intent": intent,
                    "boundary_with": confusing_intent['intent']
                }
                for sample in samples
            ]
            
            all_boundary_samples.extend(samples_with_intent)
            
            # 保存结果
            output_file = os.path.join(BOUNDARY_SAMPLES_DIR, f"{intent}_vs_{confusing_intent['intent']}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(samples_with_intent, f, ensure_ascii=False, indent=2)
            
            print(f"生成 {len(samples_with_intent)} 条边界样本，保存到: {output_file}")
    
    return all_boundary_samples


if __name__ == "__main__":
    intent = "music_recommendation"
    all_samples = generate_all_boundary_samples(intent, num_samples=10)
    
    if all_samples:
        print(f"\n总共生成 {len(all_samples)} 条边界样本")
    else:
        print("未生成边界样本")