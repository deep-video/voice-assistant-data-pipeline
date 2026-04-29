import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from src.core.seed_generator import SeedGenerator
from src.config.prompt_templates import INTENT_CONFIGS

def main():
    generator = SeedGenerator()
    intent = "music_recommendation"
    intent_config = INTENT_CONFIGS[intent]
    num_seeds = 100
    
    print("="*70)
    print("LLM驱动的表达种子生成器")
    print("="*70)
    print(f"\n意图：{intent}")
    print(f"生成数量：{num_seeds} 个种子")
    print("\n" + "="*70)
    print("开始生成种子...")
    print("="*70 + "\n")
    
    seeds = generator.generate_seeds(intent, intent_config, num_seeds)
    
    if seeds:
        print(f"\n成功生成 {len(seeds)} 个种子：\n")
        print("-"*70)
        
        for i, seed in enumerate(seeds, 1):
            user_input = seed['user_input']
            seed_type = seed.get('seed_type', 'unknown')
            
            print(f"{i}. [{seed_type}]")
            print(f"   {user_input}")
            print()
        
        print("-"*70)
        
        # 保存种子库
        generator.save_seeds(intent, seeds)
        
        print(f"\n[OK] 种子生成完成")
        print(f"保存位置：d:\\second_domain\\llm_seed_pipeline\\seed_pools\\{intent}_seeds.json")
        
        # 打印统计信息
        type_counts = {}
        for seed in seeds:
            seed_type = seed.get('seed_type', 'unknown')
            type_counts[seed_type] = type_counts.get(seed_type, 0) + 1
        
        print(f"\n种子类型分布：{type_counts}")
    else:
        print("\n[FAIL] 种子生成失败")

if __name__ == "__main__":
    main()
