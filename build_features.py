#!/usr/bin/env python
"""
预计算所有数据集特征并缓存。

运行一次即可，后续直接加载缓存，秒级启动。
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extractor import FeatureExtractor

CACHE_PATH = "features_cache"

if __name__ == "__main__":
    extractor = FeatureExtractor(data_dir=Path("."))
    result = extractor.extract_all(cache_path=CACHE_PATH)

    # 打印摘要
    qf = result["qupai_features"]
    mf = result["mode_features"]
    tf = result["tone_features"]
    rf = result["region_features"]

    print("\n" + "=" * 60)
    print("特征提取摘要")
    print("=" * 60)
    print(f"  曲牌: {len(qf)} 个")
    print(f"  宫调: {len(mf)} 个")
    print(f"  声调映射: 现代 {len(tf.get('modern_tones', {}))} 类 + 广韵 {len(tf.get('guangyun_tones', {}))} 类")
    print(f"  板块对比: {list(rf.keys())}")

    # 展示几个典型曲牌
    top_qupai = sorted(qf.items(), key=lambda x: x[1]["song_count"], reverse=True)[:5]
    print("\n  Top 5 曲牌:")
    for name, feat in top_qupai:
        print(f"    {name}: {feat['song_count']}首, "
              f"起始于{list(feat['start_pitches'].keys())[:3]}, "
              f"结束于{list(feat['end_pitches'].keys())[:3]}, "
              f"密度{feat['density']['avg_notes_per_lyric']:.2f}音/字")

    # 南词 vs 北词
    if "南詞" in rf and "北詞" in rf:
        nan = rf["南詞"]
        bei = rf["北詞"]
        print(f"\n  南词: 级进{nan['step_ratio']:.1%}, 跳进{nan['leap_ratio']:.1%}, "
              f"拖腔{nan['melisma_rate']:.1%}, 音域{nan['pitch_range']['max']-nan['pitch_range']['min']:.0f}半音")
        print(f"  北词: 级进{bei['step_ratio']:.1%}, 跳进{bei['leap_ratio']:.1%}, "
              f"拖腔{bei['melisma_rate']:.1%}, 音域{bei['pitch_range']['max']-bei['pitch_range']['min']:.0f}半音")
