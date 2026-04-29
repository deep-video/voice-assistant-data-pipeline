import openai
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME, TEMPERATURE, MAX_TOKENS
from src.config.prompt_templates import SEED_EXPANSION_PROMPT

EXPANDED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'expanded_data')
os.makedirs(EXPANDED_DATA_DIR, exist_ok=True)

class SeedExpander:
    """种子扩展器：基于单条种子生成多条变体"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_URL
        )
        self.model_name = QWEN_MODEL_NAME
    
    def expand_seed(self, seed_text: str, intent: str, intent_config: dict, num_expansions: int = 10, seed_type: str = "unknown", batch_size: int = 10, oversample_rate: float = 1.5) -> list:
        """
        基于单条种子生成变体（使用线程池并发生成）
        
        参数：
        - seed_text: 种子表达文本
        - intent: 意图名称
        - intent_config: 意图配置字典
        - num_expansions: 生成的变体数量
        - seed_type: 种子类型
        - batch_size: 每批生成的数量
        - oversample_rate: 过采样率（生成更多数据用于去重）
        
        返回：
        - 变体列表，每条包含 user_input, intent, is_seed, seed_type, source_seed
        """
        # 计算需要生成的总数（过采样）
        target_count = int(num_expansions * oversample_rate)
        num_batches = (target_count + batch_size - 1) // batch_size
        
        # 使用线程池并发生成
        all_expansions = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(num_batches):
                # 最后一批可能不足 batch_size
                current_batch_size = min(batch_size, target_count - i * batch_size)
                futures.append(
                    executor.submit(
                        self._generate_batch,
                        seed_text, intent, intent_config,
                        current_batch_size, i
                    )
                )
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_expansions.extend(result)
        
        # 去重
        unique_texts = set()
        unique_expansions = []
        for exp in all_expansions:
            if exp not in unique_texts:
                unique_texts.add(exp)
                unique_expansions.append(exp)
        
        print(f"原始生成：{len(all_expansions)} 条，去重后：{len(unique_expansions)} 条")
        
        # 添加元数据
        validated_expansions = []
        for exp in unique_expansions[:num_expansions]:
            validated_expansions.append({
                "user_input": exp,
                "intent": intent,
                "is_seed": False,
                "seed_type": seed_type,
                "source_seed": seed_text
            })
        
        return validated_expansions
    
    def _generate_batch(self, seed_text: str, intent: str, intent_config: dict, num_expansions: int, batch_idx: int) -> list:
        """生成单批变体"""
        expansion_constraints = intent_config.get("expansion_constraints", "")
        
        # 生成随机扰动因子
        random_seed = random.randint(1000, 9999)
        batch_variation = random.choice([
            "侧重句式变化（60%）",
            "侧重句式变化（60%）",
            "侧重句式变化（60%）",
            "侧重口语噪声（20%）",
            "侧重口语噪声（20%）",
            "侧重省略程度（20%）"
        ])
        
        prompt = SEED_EXPANSION_PROMPT.format(
            seed_text=seed_text,
            intent=intent,
            intent_description=intent_config.get("description", ""),
            intent_action=intent_config.get("intent_action", ""),
            intent_negative_example=intent_config.get("intent_negative_example", ""),
            num_expansions=num_expansions,
            expansion_constraints=expansion_constraints
        )
        
        # 添加批次特定指令
        prompt += f"\n\n**批次{batch_idx + 1}特殊要求**：\n- 本批{batch_variation}\n- 随机标识：{random_seed}\n- 禁止与之前批次重复"
        
        # 为每批添加temperature扰动
        batch_temperature = TEMPERATURE + random.uniform(-0.2, 0.2)
        batch_temperature = max(0.1, min(1.5, batch_temperature))  # 限制在合理范围
        
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个语言学家，专门研究人类对车载语音助手的真实说话方式。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=batch_temperature,
                    max_tokens=MAX_TOKENS,
                    extra_body={"enable_thinking": False}
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
                
                expansions = json.loads(result)
                
                if isinstance(expansions, list) and len(expansions) > 0:
                    print(f"批次 {batch_idx + 1} 生成成功，获得 {len(expansions)} 条 (temperature={batch_temperature:.2f}, 侧重={batch_variation})")
                    return expansions[:num_expansions]
                
            except Exception as e:
                print(f"批次 {batch_idx + 1} 出错 (尝试 {attempt + 1}/5): {e}")
                if attempt < 4:
                    import time
                    time.sleep(1 * (attempt + 1))
                    continue
        
        print(f"批次 {batch_idx + 1} 生成失败")
        return []
    
    def save_expansions(self, intent: str, expansions: list):
        """保存扩展数据"""
        file_path = os.path.join(EXPANDED_DATA_DIR, f"{intent}_expanded.json")
        
        # 如果文件已存在，追加数据
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        existing_data.extend(expansions)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        print(f"扩展数据已保存到：{file_path}")
        print(f"本次新增：{len(expansions)} 条")
        print(f"累计总数：{len(existing_data)} 条")
