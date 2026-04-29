import openai
import json
import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME, TEMPERATURE, MAX_TOKENS
from src.config.prompt_templates import SEED_GENERATION_PROMPT

SEED_POOL_DIR = os.path.join(os.path.dirname(__file__), 'seed_pools')
os.makedirs(SEED_POOL_DIR, exist_ok=True)

class SeedGenerator:
    """表达种子生成器：用LLM生成真实、短小、多样的用户表达"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_URL
        )
        self.model_name = QWEN_MODEL_NAME
    
    def generate_seeds(self, intent: str, intent_config: dict, num_seeds: int = 10) -> list:
        """
        生成表达种子（使用并发策略）
        
        参数：
        - intent: 意图名称
        - intent_config: 意图配置字典
        - num_seeds: 生成的种子数量
        
        返回：
        - 种子列表
        """
        # 计算分批策略
        batch_size = 30
        oversample_rate = 1.5
        target_count = int(num_seeds * oversample_rate)
        num_batches = (target_count + batch_size - 1) // batch_size
        
        print(f"目标: {num_seeds} 条，过采样目标: {target_count} 条，分 {num_batches} 批并发")
        
        # 使用线程池并发生成
        all_seeds = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(num_batches):
                current_batch_size = min(batch_size, target_count - i * batch_size)
                futures.append(
                    executor.submit(
                        self._generate_batch,
                        intent, intent_config,
                        current_batch_size, i
                    )
                )
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_seeds.extend(result)
        
        # 去重
        unique_texts = set()
        unique_seeds = []
        for seed in all_seeds:
            text = seed["user_input"]
            if text not in unique_texts:
                unique_texts.add(text)
                unique_seeds.append(seed)
        
        print(f"原始生成: {len(all_seeds)} 条，去重后: {len(unique_seeds)} 条")
        
        return unique_seeds[:num_seeds]
    
    def _generate_batch(self, intent: str, intent_config: dict, num_seeds: int, batch_idx: int) -> list:
        """生成单批种子"""
        description = intent_config.get("description", f"用户表达{intent}意图")
        intent_action = intent_config.get("intent_action", "系统提供帮助/建议")
        intent_negative_example = intent_config.get("intent_negative_example", "直接执行具体操作")
        intent_constraints = intent_config.get("constraints", "")
        
        # 随机扰动因子
        random_seed = random.randint(1000, 9999)
        batch_variation = random.choice([
            "侧重直接请求（40%）",
            "侧重情绪驱动（30%）",
            "侧重场景触发（20%）",
            "侧重模糊需求（10%）",
            "侧重省略句式（30%）",
            "侧重倒装句（20%）",
            "侧重口语化语气词（20%）"
        ])
        
        prompt = SEED_GENERATION_PROMPT.format(
            intent=intent,
            intent_description=description,
            num_seeds=num_seeds,
            intent_action=intent_action,
            intent_negative_example=intent_negative_example,
            intent_constraints=intent_constraints
        )
        
        # 添加批次特定指令
        prompt += f"\n\n**批次{batch_idx + 1}特殊要求**：\n- 本批{batch_variation}\n- 随机标识：{random_seed}\n- 禁止与其他批次重复\n- 使用不同的句式开头，避免'来点'、'推荐'等重复模式"
        
        # 温度扰动
        batch_temperature = TEMPERATURE + random.uniform(-0.2, 0.3)
        batch_temperature = max(0.5, min(1.2, batch_temperature))
        
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
                
                seeds = json.loads(result)
                
                if isinstance(seeds, list) and len(seeds) > 0:
                    validated_seeds = []
                    for seed in seeds:
                        if isinstance(seed, dict) and "text" in seed:
                            validated_seeds.append({
                                "user_input": seed["text"],
                                "intent": intent,
                                "is_seed": True,
                                "seed_type": seed.get("type", "direct_request")
                            })
                        elif isinstance(seed, str):
                            validated_seeds.append({
                                "user_input": seed,
                                "intent": intent,
                                "is_seed": True,
                                "seed_type": "direct_request"
                            })
                    
                    print(f"批次 {batch_idx + 1} 生成成功，获得 {len(validated_seeds)} 条 (temperature={batch_temperature:.2f}, 侧重={batch_variation})")
                    return validated_seeds
                
            except Exception as e:
                print(f"批次 {batch_idx + 1} 失败 (尝试 {attempt + 1}/5): {e}")
                if attempt < 4:
                    import time
                    time.sleep(1 * (attempt + 1))
                    continue
        
        print(f"批次 {batch_idx + 1} 生成失败")
        return []
    
    def save_seeds(self, intent: str, seeds: list):
        """保存种子到种子库"""
        file_path = os.path.join(SEED_POOL_DIR, f"{intent}_seeds.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(seeds, f, ensure_ascii=False, indent=2)
        print(f"种子库已保存到：{file_path}")
    
    def load_seeds(self, intent: str) -> list:
        """从种子库加载种子"""
        file_path = os.path.join(SEED_POOL_DIR, f"{intent}_seeds.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
