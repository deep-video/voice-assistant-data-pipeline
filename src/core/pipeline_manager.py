"""
数据生成 Pipeline 管理器

流程：种子生成 → 种子扩展 → 边界样本生成 → 质量过滤
"""

import json
import os
import sys
import math
import argparse
from typing import Dict, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# 设置控制台编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config.api_config import QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL_NAME
from src.config.intent_descriptions import INTENT_DESCRIPTIONS, ALL_INTENTS
from src.core.seed_generator import SeedGenerator
from src.core.seed_expander import SeedExpander
from src.core.generate_boundary_samples import generate_all_boundary_samples, load_confusing_intents
# 使用本地模型版本的质量评估（而非 API 版本）
from src.evaluation.evaluate_data_quality import evaluate_data as evaluate_data_quality


class PipelineConfig:
    """
    Pipeline 配置 - 可快速调整数据比例
    
    修改说明：
    - SEED_RATIO: 种子数据占比（建议 5-15%）
    - EXPAND_RATIO: 扩展数据占比（建议 70-85%）
    - BOUNDARY_RATIO: 边界数据占比（建议 5-15%）
    - 三个比例之和必须等于 1.0
    """
    # 数据比例（可调整）
    SEED_RATIO = 0.20      # 种子占比 20%
    EXPAND_RATIO = 0.70    # 扩展占比 70%
    BOUNDARY_RATIO = 0.10  # 边界占比 10%
    
    # 质量过滤预估通过率（根据历史数据调整）
    ESTIMATED_PASS_RATE = 0.85


class DataPipeline:
    """数据生成 Pipeline 管理器"""
    
    def __init__(self, intent: str, config: PipelineConfig = None):
        self.intent = intent
        self.intent_desc = INTENT_DESCRIPTIONS[intent]
        self.config = config or PipelineConfig()
        
        # 初始化各模块
        self.seed_generator = SeedGenerator()
        self.seed_expander = SeedExpander()
        
        # 输出目录（都在 core 目录下）
        self.seed_pool_dir = os.path.join(os.path.dirname(__file__), "seed_pools")
        self.expanded_data_dir = os.path.join(os.path.dirname(__file__), "expanded_data")
        self.boundary_samples_dir = os.path.join(os.path.dirname(__file__), "boundary_samples")
        self.quality_evaluation_dir = os.path.join(os.path.dirname(__file__), "quality_evaluation")
        
        # 确保目录存在
        os.makedirs(self.seed_pool_dir, exist_ok=True)
        os.makedirs(self.expanded_data_dir, exist_ok=True)
        os.makedirs(self.boundary_samples_dir, exist_ok=True)
        os.makedirs(self.quality_evaluation_dir, exist_ok=True)
    
    def calculate_parameters(self, target_count: int) -> Dict:
        """
        根据目标数量自动计算各阶段参数
        
        参数：
        - target_count: 最终想要的数据量
        
        返回：
        - dict: 包含 num_seeds, num_expansions_per_seed, num_boundary_samples_per_intent
        """
        # 考虑质量过滤损耗，计算需要生成的总量
        total_needed = math.ceil(target_count / self.config.ESTIMATED_PASS_RATE)
        
        # 按比例分配
        num_seeds = math.ceil(total_needed * self.config.SEED_RATIO)
        num_expanded = math.ceil(total_needed * self.config.EXPAND_RATIO)
        num_boundary = math.ceil(total_needed * self.config.BOUNDARY_RATIO)
        
        # 计算每个种子需要扩展多少条
        num_expansions_per_seed = math.ceil(num_expanded / num_seeds)
        
        # 计算每个混淆意图需要生成多少条边界样本
        confusing_intents = load_confusing_intents(self.intent)
        num_confusing = max(len(confusing_intents), 1)  # 至少1个
        num_boundary_per_intent = math.ceil(num_boundary / num_confusing)
        
        return {
            "target_count": target_count,
            "estimated_pass_rate": self.config.ESTIMATED_PASS_RATE,
            "total_to_generate": total_needed,
            "num_seeds": num_seeds,
            "num_expanded": num_expanded,
            "num_boundary": num_boundary,
            "num_expansions_per_seed": num_expansions_per_seed,
            "num_confusing_intents": num_confusing,
            "num_boundary_per_intent": num_boundary_per_intent,
            "ratios": {
                "seed": self.config.SEED_RATIO,
                "expand": self.config.EXPAND_RATIO,
                "boundary": self.config.BOUNDARY_RATIO
            }
        }
    
    def run_pipeline(self, 
                     target_count: int = 1000,
                     auto_calculate: bool = True,
                     num_seeds: int = None,
                     num_expansions_per_seed: int = None,
                     num_boundary_samples: int = None,
                     run_quality_eval: bool = True):
        """
        运行完整 Pipeline
        
        参数：
        - target_count: 最终想要的数据量（auto_calculate=True 时生效）
        - auto_calculate: 是否自动计算参数（默认 True）
        - num_seeds: 手动设置种子数量（auto_calculate=False 时生效）
        - num_expansions_per_seed: 手动设置每个种子扩展数量
        - num_boundary_samples: 手动设置每个混淆意图边界样本数量
        - run_quality_eval: 是否运行质量过滤
        
        返回：
        - 保留的数据列表
        """
        if auto_calculate:
            params = self.calculate_parameters(target_count)
            num_seeds = params["num_seeds"]
            num_expansions_per_seed = params["num_expansions_per_seed"]
            num_boundary_samples = params["num_boundary_per_intent"]
            
            print("=" * 70)
            print(f"目标数量: {target_count} 条")
            print(f"预估通过率: {self.config.ESTIMATED_PASS_RATE:.0%}")
            print(f"需要生成总量: {params['total_to_generate']} 条")
            print(f"数据比例: 种子 {params['ratios']['seed']:.0%} / 扩展 {params['ratios']['expand']:.0%} / 边界 {params['ratios']['boundary']:.0%}")
            print(f"自动计算参数:")
            print(f"  - 种子数量: {num_seeds}")
            print(f"  - 每个种子扩展: {num_expansions_per_seed} 条")
            print(f"  - 混淆意图数: {params['num_confusing_intents']} 个")
            print(f"  - 每个混淆意图边界样本: {num_boundary_samples} 条")
            print(f"  - 预估扩展数据: {params['num_expanded']} 条")
            print(f"  - 预估边界数据: {params['num_boundary']} 条")
            print("=" * 70)
        else:
            print("=" * 70)
            print(f"手动模式 - 意图: {self.intent}")
            print(f"  - 种子数量: {num_seeds}")
            print(f"  - 每个种子扩展: {num_expansions_per_seed} 条")
            print(f"  - 每个混淆意图边界样本: {num_boundary_samples} 条")
            print("=" * 70)
        
        print(f"\n开始运行数据生成 Pipeline - 意图: {self.intent}")
        print("=" * 70)
        
        # 步骤1: 生成种子
        print("\n[步骤1/4] 生成种子...")
        seeds = self._generate_seeds(num_seeds)
        
        # 步骤2: 扩展种子
        print("\n[步骤2/4] 扩展种子...")
        expanded_data = self._expand_seeds(num_expansions_per_seed)
        
        # 步骤3: 生成边界样本（自动使用大模型判断混淆意图）
        print("\n[步骤3/4] 生成边界样本...")
        boundary_samples = generate_all_boundary_samples(self.intent, num_boundary_samples)
        
        # 步骤4: 质量过滤
        if run_quality_eval:
            print("\n[步骤4/4] 质量过滤...")
            keep_data = self._evaluate_quality(expanded_data, boundary_samples)
        else:
            print("\n[跳过] 质量过滤")
            keep_data = expanded_data + boundary_samples
        
        print("\n" + "=" * 70)
        print("Pipeline 完成！")
        print("=" * 70)
        
        return keep_data
    
    def _generate_seeds(self, num_seeds: int) -> List[Dict]:
        """生成种子"""
        intent_config = {
            "description": self.intent_desc['description'],
            "intent_action": self.intent_desc['intent_action'],
            "intent_negative_example": self.intent_desc['intent_negative_example']
        }
        
        seeds = self.seed_generator.generate_seeds(
            intent=self.intent,
            intent_config=intent_config,
            num_seeds=num_seeds
        )
        
        # 保存种子
        output_file = os.path.join(self.seed_pool_dir, f"{self.intent}_seeds.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(seeds, f, ensure_ascii=False, indent=2)
        
        print(f"生成 {len(seeds)} 条种子，保存到: {output_file}")
        return seeds
    
    def _expand_seeds(self, num_expansions: int) -> List[Dict]:
        """扩展种子"""
        # 从种子池加载种子
        seed_pool_file = os.path.join(self.seed_pool_dir, f"{self.intent}_seeds.json")
        if not os.path.exists(seed_pool_file):
            print(f"种子池文件不存在: {seed_pool_file}")
            return []
        
        with open(seed_pool_file, "r", encoding="utf-8") as f:
            seed_pool = json.load(f)
        
        intent_config = {
            "description": self.intent_desc['description'],
            "intent_action": self.intent_desc['intent_action'],
            "intent_negative_example": self.intent_desc['intent_negative_example']
        }
        
        all_expanded_data = []
        
        # 对所有种子进行扩展
        for i, seed in enumerate(seed_pool):
            seed_text = seed.get("user_input") or seed.get("text", "")
            seed_type = seed.get("type", "unknown")
            
            print(f"\n扩展种子 {i+1}/{len(seed_pool)}: {seed_text}")
            
            expanded_data = self.seed_expander.expand_seed(
                seed_text=seed_text,
                intent=self.intent,
                intent_config=intent_config,
                num_expansions=num_expansions,
                seed_type=seed_type
            )
            
            all_expanded_data.extend(expanded_data)
        
        # 保存扩展数据
        output_file = os.path.join(self.expanded_data_dir, f"{self.intent}_expanded.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_expanded_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n生成 {len(all_expanded_data)} 条扩展数据，保存到: {output_file}")
        return all_expanded_data
    
    def _evaluate_quality(self, expanded_data: List[Dict], boundary_samples: List[Dict]):
        """质量过滤"""
        # 合并数据
        all_data = expanded_data + boundary_samples
        
        if not all_data:
            print("没有数据需要评估")
            return
        
        print(f"开始评估 {len(all_data)} 条数据...")
        
        # 运行质量评估
        keep_data, discard_data = evaluate_data_quality(all_data)
        
        # 保存结果
        keep_file = os.path.join(self.quality_evaluation_dir, f"{self.intent}_keep.json")
        discard_file = os.path.join(self.quality_evaluation_dir, f"{self.intent}_discard.json")
        
        with open(keep_file, "w", encoding="utf-8") as f:
            json.dump(keep_data, f, ensure_ascii=False, indent=2)
        
        with open(discard_file, "w", encoding="utf-8") as f:
            json.dump(discard_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n质量过滤结果:")
        print(f"  保留: {len(keep_data)} 条 -> {keep_file}")
        print(f"  丢弃: {len(discard_data)} 条 -> {discard_file}")
        
        return keep_data


def merge_all_intents_data(output_dir: str = None):
    """合并所有意图的保留数据到一个JSON文件"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'final_dataset')
    
    os.makedirs(output_dir, exist_ok=True)
    
    quality_evaluation_dir = os.path.join(os.path.dirname(__file__), 'quality_evaluation')
    
    all_data = []
    
    for intent in ALL_INTENTS:
        keep_file = os.path.join(quality_evaluation_dir, f"{intent}_keep.json")
        
        if not os.path.exists(keep_file):
            print(f"警告: {intent} 的保留数据文件不存在")
            continue
        
        with open(keep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # 确保每条数据都有 intent 字段
            for item in data:
                if "intent" not in item:
                    item["intent"] = intent
            
            all_data.extend(data)
            print(f"加载 {intent}: {len(data)} 条")
    
    # 保存合并后的数据
    output_file = os.path.join(output_dir, "all_intents_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n合并完成！")
    print(f"总数据量: {len(all_data)} 条")
    print(f"保存到: {output_file}")
    
    # 统计各意图分布
    intent_counts = {}
    for item in all_data:
        intent = item.get("intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n意图分布:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(all_data) * 100
        print(f"  {intent}: {count} 条 ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据生成 Pipeline")
    parser.add_argument("--intent", type=str, default=None, help="指定意图名称，不指定则生成所有意图的数据")
    parser.add_argument("--target", type=int, default=1000, help="每个意图的目标数据量（默认1000）")
    parser.add_argument("--no-eval", action="store_true", help="跳过质量评估")
    parser.add_argument("--merge-only", action="store_true", help="仅合并已有数据")
    args = parser.parse_args()
    
    if args.merge_only:
        merge_all_intents_data()
    elif args.intent:
        # 生成指定意图的数据
        pipeline = DataPipeline(args.intent)
        pipeline.run_pipeline(
            target_count=args.target,
            auto_calculate=True,
            run_quality_eval=not args.no_eval
        )
    else:
        # 生成所有意图的数据
        for intent in ALL_INTENTS:
            print(f"\n{'='*70}")
            print(f"开始处理意图: {intent}")
            print(f"{'='*70}")
            
            pipeline = DataPipeline(intent)
            pipeline.run_pipeline(
                target_count=args.target,
                auto_calculate=True,
                run_quality_eval=not args.no_eval
            )
        
        # 所有意图生成完成后，合并数据
        merge_all_intents_data()
