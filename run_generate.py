#!/usr/bin/env python
"""端到端生成测试：加载 → 检索 → 生成 → 双格式输出"""

import sys, os, json, re, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(".env")

import pandas as pd
from openai import OpenAI

from src.gongche_vocab import parse_compact_gongche
from src.musicxml_writer import gongche_to_musicxml

t_total = time.time()

# ====== Step 1: 检索 Few-Shot 样例 ======
print("=" * 50)
print("Step 1: 检索 Few-Shot 样例")
print("=" * 50)

songs_df = pd.read_csv("FINAL_SONGS.csv", encoding="utf-8")
notes_df = pd.read_csv("FINAL_NOTES.csv", encoding="utf-8", low_memory=False)

fengshi_ids = songs_df[songs_df["qupai_brackets_removed"] == "奉時春"]["polyu_id"].tolist()
print(f"奉時春 歌曲数: {len(fengshi_ids)}")

example_id = fengshi_ids[0]
example_notes = notes_df[notes_df["polyu_id"].astype(str) == str(example_id)]
example_notes = example_notes.sort_values(["beat_id", "lyric_id", "gongche_id"])

# 构建 few-shot 文本
groups = defaultdict(list)
for _, r in example_notes.iterrows():
    key = (r["line_id"], r["lyric_id"])
    groups[key].append((r["lyric"], r["gongche"], float(r["duration"])))

parts = []
for key in sorted(groups.keys()):
    items = groups[key]
    lyric_char = str(items[0][0])
    gongches = []
    for gc, dur in [(i[1], i[2]) for i in items]:
        m = ""
        if abs(dur - 0.5) < 0.01:
            m = "(O)"
        elif abs(dur - 0.125) < 0.01:
            m = "(x)"
        gongches.append(f"{gc}{m}")
    parts.append(f"{lyric_char}[{' '.join(gongches)}]")

fewshot_text = " ".join(parts)
example_lyrics = "".join(
    str(groups[k][0][0]) for k in sorted(groups.keys())
)
print(f"样例歌词 ({len(example_lyrics)}字): {example_lyrics}")
print(f"样例工尺 (前150): {fewshot_text[:150]}...")
print()

# ====== Step 2: API 生成 ======
print("=" * 50)
print("Step 2: 调用 DeepSeek 生成")
print("=" * 50)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """你是精通《九宫大成南北词宫谱》的中国古典音乐大师。
工尺音阶(低到高): 合 四 一 上(宫音) 尺 工 凡 六 五 乙 仩 伬
作曲规则:
- 一字可对1-3个工尺音符(拖腔)
- 南词柔婉级进为主,北词刚健跳进为主
- 起音收音稳定,同一曲牌旋律骨架相似
时长标记: (O)=二分长音 (x)=八分音符 默认无标记=四分音符
输出格式(严格): 旋律: <字>[<工尺> <工尺>...] <字>[<工尺>...] ...
只输出一行旋律,绝对不要输出任何解释、分析或其他内容。"""

NEW_LYRICS = "春風拂柳燕歸來"

user_prompt = f"""## 参考样例 (曲牌《奉時春》, 南詞·仙呂宮引)
歌词: {example_lyrics}
旋律: {fewshot_text}

## 生成任务
曲牌: 奉時春
板块: 南詞
宫调: 仙呂宮引
新歌词: {NEW_LYRICS}
请生成工尺谱旋律:"""

t_api = time.time()
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.8,
    max_tokens=300,
)
api_time = time.time() - t_api
raw_response = resp.choices[0].message.content
print(f"API 耗时: {api_time:.1f}s")
print(f"Tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
print()

# ====== Step 3: 解析 & 输出 ======
print("=" * 50)
print("Step 3: 解析 & 保存工尺谱文本")
print("=" * 50)

print(f"DeepSeek 原始响应:\n{raw_response}\n")

matches = re.findall(r"(\S+)\[([^\]]+)\]", raw_response)
print(f"解析到 {len(matches)} 个字-音组:")

out_dir = Path("experiments/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 保存工尺谱文本
txt_path = out_dir / "test_generation.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"曲牌: 奉時春\n")
    f.write(f"板块: 南詞\n")
    f.write(f"宫调: 仙呂宮引\n")
    f.write(f"\n")
    f.write(f"歌词: {NEW_LYRICS}\n")
    f.write(f"{raw_response}\n")
    f.write(f"\n")
    f.write(f"Token 用量: {resp.usage.total_tokens}\n")
    f.write(f"生成耗时: {api_time:.1f}s\n")
print(f"工尺谱文本: {txt_path}")

# ====== Step 4: MusicXML ======
print()
print("=" * 50)
print("Step 4: 生成 MusicXML")
print("=" * 50)

gc_groups = parse_compact_gongche(raw_response)
if gc_groups:
    for g in gc_groups:
        gcs = [n.gongche for n in g.notes]
        print(f"  {g.lyric}: {gcs} ({len(g.notes)}音)")

    xml_path = out_dir / "test_generation.musicxml"
    gongche_to_musicxml(
        lyric_groups=gc_groups,
        output_path=str(xml_path),
        title=f"AI生成·奉時春·{NEW_LYRICS}",
        work_title="九宫大成·AI生成",
        qupai="奉時春",
    )
    print(f"\nMusicXML: {xml_path} ({os.path.getsize(str(xml_path))} bytes)")
    print(f"  -> 可用 MuseScore 打开播放")
else:
    print("解析失败，跳过 MusicXML 生成")

# ====== 保存元数据 ======
meta = {
    "qupai": "奉時春",
    "lyrics": NEW_LYRICS,
    "raw_response": raw_response,
    "prompt_tokens": resp.usage.prompt_tokens,
    "completion_tokens": resp.usage.completion_tokens,
    "api_time": round(api_time, 2),
    "total_time": round(time.time() - t_total, 2),
}
json.dump(
    meta,
    (out_dir / "test_generation.meta.json").open("w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)

print()
print(f"===== 全部完成! 总耗时 {time.time() - t_total:.1f}s =====")
