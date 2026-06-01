#!/usr/bin/env python
"""
九宫大成诗词音乐生成系统 — 测试/演示入口。

演示流程：
1. 加载数据集
2. 构建曲牌索引
3. 测试生成（如设置 API key）
4. 测试 MusicXML 转换
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_loader import get_dataset
from src.gongche_vocab import (
    parse_compact_gongche, format_lyric_gongche_pairs,
    parse_llm_response, GONGCHE_TO_PITCH, GONGCHE_TO_MIDI,
    melody_similarity,
)
from src.musicxml_writer import gongche_to_musicxml, song_to_musicxml
from src.qupai_index import QupaiIndex
from src.prompt_templates import PromptBuilder, get_experiment_prompts
from src.generator import QupaiMusicGenerator


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)

    ds = get_dataset()
    print(f"  歌曲总数: {len(ds)}")
    print(f"  宫调数: {len(ds.get_mode_list())}")
    regions = ds.get_region_distribution()
    print(f"  板块分布: 南词={regions.get('南詞', 0)}, 北词={regions.get('北詞', 0)}, 合套={regions.get('合套', 0)}")

    # 测试单曲加载
    song = ds.get_song("287.1")
    print(f"  示例歌曲 287.1:")
    print(f"    曲牌: {song.qupai}")
    print(f"    来源: {song.source}")
    print(f"    宫调: {song.volume_mode}")
    print(f"    板块: {song.volume_region}")
    print(f"    歌词: {song.lyrics_text}")
    print(f"    音符数: {len(song.notes)}")
    print(f"    歌词分组: {len(song.lyric_note_groups)}")
    print(f"    音符/字: {song.notes_per_lyric:.2f}")
    print(f"    拖腔比例: {song.melisma_ratio:.2%}")
    print(f"    工尺序列 (前20): {' '.join(n.gongche for n in song.notes[:20])}")

    return ds


def test_gongche_vocab():
    """测试工尺谱词表"""
    print("\n" + "=" * 60)
    print("测试 2: 工尺谱词表")
    print("=" * 60)

    # 测试解析
    test_input = "旋律: 風[工] 和[尺 工] 日[上 尺] 麗[工 五] 布[五 六] 艷[工] 陽[尺 工(O)]"
    groups, text = parse_llm_response(test_input)
    print(f"  输入: {test_input}")
    print(f"  解析结果: {len(groups)} 组")
    for g in groups:
        gcs = " ".join(n.gongche for n in g.notes)
        print(f"    {g.lyric}: [{gcs}] ({len(g.notes)}音)")

    # 测试格式化
    formatted = format_lyric_gongche_pairs(groups)
    print(f"  重新格式化: {formatted}")

    # 测试音高映射
    print(f"  工尺→MIDI: 上→{GONGCHE_TO_MIDI['上']}, 尺→{GONGCHE_TO_MIDI['尺']}, 工→{GONGCHE_TO_MIDI['工']}")

    return groups


def test_musicxml_export(dataset, groups):
    """测试 MusicXML 导出"""
    print("\n" + "=" * 60)
    print("测试 3: MusicXML 导出")
    print("=" * 60)

    output_dir = dataset.data_dir / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出真实歌曲
    song = dataset.get_song("287.1")
    real_path = output_dir / "test_real_song.musicxml"
    gongche_to_musicxml(
        lyric_groups=song.lyric_note_groups,
        output_path=str(real_path),
        title=f"{song.polyu_id}:{song.source}《{song.qupai_bare}》",
        work_title="九宫大成·原始数据",
        qupai=song.qupai_bare,
        source=song.source,
    )
    print(f"  真实歌曲 MusicXML: {real_path} ({os.path.getsize(str(real_path))} bytes)")

    # 导出生成旋律
    gen_path = output_dir / "test_generated.musicxml"
    gongche_to_musicxml(
        lyric_groups=groups,
        output_path=str(gen_path),
        title="AI生成·奉時春",
        work_title="九宫大成·AI生成",
        qupai="奉時春",
    )
    print(f"  生成旋律 MusicXML: {gen_path} ({os.path.getsize(str(gen_path))} bytes)")

    return str(gen_path)


def test_qupai_index(dataset):
    """测试曲牌索引"""
    print("\n" + "=" * 60)
    print("测试 4: 曲牌索引")
    print("=" * 60)

    idx = QupaiIndex(dataset)
    idx.build(min_song_count=1)
    print(f"  索引曲牌数: {len(idx)}")

    # 测试检索
    examples = idx.retrieve_examples("奉時春", "風和日麗", n=3)
    print(f"  奉時春 样例检索: {len(examples)} 个")
    for ex in examples:
        print(f"    {ex['polyu_id']}: {len(ex['lyrics'])}字歌词, "
              f"{ex['note_count']}音符, score={ex['score']:.3f}")

    # 测试宫调查询
    top_qupai = idx.get_qupai_by_mode("仙呂宮引", top_k=3)
    print(f"  仙呂宮引 top qupai: {top_qupai}")

    return idx


def test_prompt_building(index):
    """测试 Prompt 构建"""
    print("\n" + "=" * 60)
    print("测试 5: Prompt 构建")
    print("=" * 60)

    examples = index.retrieve_examples("奉時春", "風和日麗", n=2)

    for strategy in ["zeroshot", "fewshot", "fewshot_tones"]:
        builder = PromptBuilder(strategy=strategy, annotation="minimal")
        system, user = builder.build(
            qupai="奉時春",
            lyrics="風和日麗",
            examples=examples,
        )
        print(f"\n  --- 策略: {strategy} ---")
        print(f"  System prompt: {len(system)} 字符")
        print(f"  User prompt: {len(user)} 字符")
        print(f"  User prompt 前 200 字符: {user[:200]}...")


def test_generator_api(index, dataset):
    """测试生成器（需要 API key）"""
    print("\n" + "=" * 60)
    print("测试 6: 生成器就绪检查")
    print("=" * 60)

    config = Config()
    gen = QupaiMusicGenerator(config, dataset)
    gen.index = index  # 复用已有的索引
    # 不初始化 client，只测试非 API 部分

    status = gen.check_readiness()
    print(f"  数据集: {'OK' if status['dataset_loaded'] else 'FAIL'}")
    print(f"  曲牌索引: {'OK' if status['index_built'] else 'FAIL'} "
          f"({status.get('qupai_count', 0)} 曲牌)")
    print(f"  API 就绪: {'OK' if status['api_connected'] else '未连接 (设置 DEEPSEEK_API_KEY 后可用)'}")

    if not status['api_connected']:
        print("\n  [提示] 设置环境变量 DEEPSEEK_API_KEY 后可以运行完整生成测试:")
        print("    export DEEPSEEK_API_KEY=sk-xxxx")
        print("    python src/generator.py --qupai 奉時春 --lyrics 風和日麗")

    return gen


def main():
    """运行所有测试"""
    print("九宫大成诗词音乐生成系统 — 测试套件\n")

    # 1. 数据加载
    ds = test_data_loading()

    # 2. 工尺谱词表
    groups = test_gongche_vocab()

    # 3. MusicXML 导出
    xml_path = test_musicxml_export(ds, groups)

    # 4. 曲牌索引
    idx = test_qupai_index(ds)

    # 5. Prompt 构建
    test_prompt_building(idx)

    # 6. 生成器检查
    test_generator_api(idx, ds)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print(f"MusicXML 文件: {xml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
