"""
检查种子数据中的重复和相似项
"""

import json
import os
import sys
from collections import Counter

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def check_duplicates_and_similarity(file_path: str):
    """检查重复和相似数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    
    texts = [s["user_input"] for s in seeds]
    
    # 1. 检查完全重复
    text_counts = Counter(texts)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    
    print("=" * 70)
    print(f"文件: {file_path}")
    print(f"总数量: {len(seeds)} 条")
    print("=" * 70)
    
    if duplicates:
        print(f"\n完全重复项 ({len(duplicates)} 组):")
        print("-" * 70)
        for text, count in duplicates.items():
            print(f"  重复 {count} 次: {text}")
    else:
        print("\n无完全重复项")
    
    # 2. 检查高度相似（基于关键词重叠）
    print("\n\n高度相似组（基于关键词重叠）:")
    print("-" * 70)
    
    # 提取关键词模式
    patterns = {}
    for text in texts:
        # 提取核心模式
        if "来点" in text:
            pattern = text.replace("来点", "XXX")
        elif "推荐" in text:
            pattern = text.replace("推荐", "XXX")
        elif "给点" in text:
            pattern = text.replace("给点", "XXX")
        elif "有啥" in text:
            pattern = text.replace("有啥", "XXX")
        else:
            pattern = text
        
        # 进一步简化
        for word in ["的", "了", "吗", "呢", "啊", "！", "？", "…", "～"]:
            pattern = pattern.replace(word, "")
        
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(text)
    
    # 找出相似组
    similar_groups = [(pattern, items) for pattern, items in patterns.items() if len(items) > 1]
    similar_groups.sort(key=lambda x: len(x[1]), reverse=True)
    
    for pattern, items in similar_groups[:20]:  # 只显示前20组
        print(f"\n模式: {pattern}")
        for item in items:
            print(f"  - {item}")
    
    # 3. 统计句式分布
    print("\n\n句式分布:")
    print("-" * 70)
    sentence_patterns = {
        "来点...": sum(1 for t in texts if "来点" in t),
        "推荐...": sum(1 for t in texts if "推荐" in t),
        "给点...": sum(1 for t in texts if "给点" in t),
        "有啥...": sum(1 for t in texts if "有啥" in t),
        "其他": sum(1 for t in texts if all(w not in t for w in ["来点", "推荐", "给点", "有啥"]))
    }
    
    for pattern, count in sentence_patterns.items():
        percentage = count / len(texts) * 100
        print(f"  {pattern}: {count} 条 ({percentage:.1f}%)")


if __name__ == "__main__":
    file_path = os.path.join(PROJECT_ROOT, "src", "core", "seed_pools", "music_recommendation_test_100.json")
    check_duplicates_and_similarity(file_path)
