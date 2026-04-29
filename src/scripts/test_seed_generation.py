"""
测试种子生成：一次性生成100条数据，查看效果
"""

import json
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.config.intent_descriptions import INTENT_DESCRIPTIONS
from src.core.seed_generator import SeedGenerator


def test_seed_generation(intent: str, num_seeds: int = 100):
    """测试种子生成"""
    print(f"=" * 70)
    print(f"测试意图: {intent}")
    print(f"生成数量: {num_seeds}")
    print(f"=" * 70)
    
    # 初始化生成器
    generator = SeedGenerator()
    
    # 准备意图配置
    intent_config = {
        "description": INTENT_DESCRIPTIONS[intent]['description'],
        "intent_action": INTENT_DESCRIPTIONS[intent]['intent_action'],
        "intent_negative_example": INTENT_DESCRIPTIONS[intent]['intent_negative_example']
    }
    
    # 生成种子
    print("\n开始生成...")
    seeds = generator.generate_seeds(
        intent=intent,
        intent_config=intent_config,
        num_seeds=num_seeds
    )
    
    if not seeds:
        print("生成失败！")
        return
    
    print(f"\n成功生成 {len(seeds)} 条种子\n")
    
    # 打印前20条查看效果
    print("前20条种子：")
    print("-" * 70)
    for i, seed in enumerate(seeds[:20], 1):
        print(f"{i:3d}. [{seed.get('seed_type', 'unknown')}] {seed['user_input']}")
    
    # 统计类型分布
    type_counts = {}
    for seed in seeds:
        seed_type = seed.get('seed_type', 'unknown')
        type_counts[seed_type] = type_counts.get(seed_type, 0) + 1
    
    print(f"\n类型分布：")
    print("-" * 70)
    for seed_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(seeds) * 100
        print(f"  {seed_type}: {count} 条 ({percentage:.1f}%)")
    
    # 保存结果
    output_file = os.path.join(PROJECT_ROOT, "src", "core", "seed_pools", f"{intent}_test_100.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果保存到: {output_file}")
    print(f"=" * 70)


if __name__ == "__main__":
    # 测试音乐推荐意图
    test_seed_generation("music_recommendation", num_seeds=100)
