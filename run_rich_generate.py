#!/usr/bin/env python
"""Rich Prompt 生成测试：加载全部特征 + 检索样例 → DeepSeek → 工尺谱 + MusicXML"""

import sys, os, json, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(".env")

import pandas as pd
from openai import OpenAI

from src.rich_prompt import RichPromptBuilder
from src.gongche_vocab import parse_compact_gongche
from src.musicxml_writer import gongche_to_musicxml

t_total = time.time()

# ====== Step 1: 加载特征缓存 ======
print("=" * 60)
print("Step 1: 加载特征缓存")
print("=" * 60)

with open("features_cache.json", "r", encoding="utf-8") as f:
    features = json.load(f)
builder = RichPromptBuilder(features)
print(f"  曲牌: {len(features['qupai_features'])} 个")
print(f"  宫调: {len(features['mode_features'])} 个")
print(f"  声调: {len(features['tone_features']['guangyun_tones'])} 类")

# ====== Step 2: 检索样例 ======
QUPAI = "小桃紅"   # 数据集有26首，够做 few-shot
NEW_LYRICS = """小院春寒
梨花半落胭脂雨
绿窗朱户
燕绕秋千柱
心事轻梳
欲语还羞住
风起处
落红无数
谁共斜阳暮"""

print("\n" + "=" * 60)
print(f"Step 2: 检索《{QUPAI}》样例")
print("=" * 60)

notes_df = pd.read_csv("FINAL_NOTES.csv", encoding="utf-8", low_memory=False)
songs_df = pd.read_csv("FINAL_SONGS.csv", encoding="utf-8")

qupai_ids = songs_df[songs_df["qupai_brackets_removed"] == QUPAI]["polyu_id"].tolist()
print(f"  《{QUPAI}》在数据集中有 {len(qupai_ids)} 首")

# 取样例
from collections import defaultdict

examples = []
for pid in qupai_ids[:2]:
    song_notes = notes_df[notes_df["polyu_id"].astype(str) == str(pid)]
    song_notes = song_notes.sort_values(["beat_id", "lyric_id", "gongche_id"])

    groups = defaultdict(list)
    for _, r in song_notes.iterrows():
        key = (r["line_id"], r["lyric_id"])
        groups[key].append({
            "lyric": str(r["lyric"]),
            "gongche": str(r["gongche"]),
            "gongche_pitch": float(r["gongche_pitch"]),
            "duration": float(r["duration"]),
            "is_melisma": int(r["multi_rhythm_num"]) > 0,
        })

    # 构建 LyricNoteGroup 兼容结构
    from src.data_loader import NoteEvent, LyricNoteGroup
    lyric_groups = []
    for key in sorted(groups.keys()):
        items = groups[key]
        notes = []
        for item in items:
            notes.append(NoteEvent(
                lyric=item["lyric"],
                lyric_modern_tone=0,
                lyric_is_entering=False,
                lyric_guangyun_tone="",
                gongche=item["gongche"],
                gongche_pitch=item["gongche_pitch"],
                duration=item["duration"],
                beat_id=0, line_id=0, lyric_id=0,
                is_melisma=item["is_melisma"],
                melisma_id=0, rhythm="",
            ))
        lyric_groups.append(LyricNoteGroup(
            lyric=items[0]["lyric"],
            lyric_modern_tone=0,
            lyric_is_entering=False,
            lyric_guangyun_tone="",
            notes=notes,
        ))

    lyrics_text = "".join(str(g.lyric) for g in lyric_groups)
    examples.append({
        "polyu_id": str(pid),
        "lyrics": lyrics_text,
        "lyric_note_groups": lyric_groups,
        "note_count": len(song_notes),
    })
    print(f"  样例 {pid}: {len(lyrics_text)}字歌词, {len(song_notes)}音符")

# ====== Step 3: 构建 Rich Prompt ======
print("\n" + "=" * 60)
print("Step 3: 构建 Rich Prompt")
print("=" * 60)

system_prompt, user_prompt = builder.build(
    qupai=QUPAI,
    lyrics=NEW_LYRICS,
    examples=examples,
)
print(f"  System: {len(system_prompt)} 字符")
print(f"  User: {len(user_prompt)} 字符")
print(f"\n--- System Prompt (前600字符) ---")
print(system_prompt[:600])
print(f"\n--- User Prompt (前600字符) ---")
print(user_prompt[:600])

# ====== Step 4: API 调用 ======
print("\n" + "=" * 60)
print("Step 4: 调用 DeepSeek 生成")
print("=" * 60)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

t_api = time.time()
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.8,
    max_tokens=400,
)
api_time = time.time() - t_api
raw = resp.choices[0].message.content
print(f"  API 耗时: {api_time:.1f}s")
print(f"  Tokens: prompt={resp.usage.prompt_tokens}, "
      f"completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

# ====== Step 5: 解析 & 保存 ======
print("\n" + "=" * 60)
print("Step 5: 解析 & 生成输出文件")
print("=" * 60)

print(f"\nDeepSeek 响应:\n{raw}\n")

# Multi-line aware display
lyric_lines = [l.strip() for l in NEW_LYRICS.split("\n") if l.strip()]
is_multi = len(lyric_lines) > 1

gc_groups = parse_compact_gongche(raw)
out_dir = Path("experiments/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

if gc_groups:
    if is_multi:
        # Show line-by-line
        print(f"解析到 {len(gc_groups)} 组 ({len(lyric_lines)} 行):")
        char_idx = 0
        for line_no, line_text in enumerate(lyric_lines):
            line_groups = gc_groups[char_idx:char_idx + len(line_text)]
            char_idx += len(line_text)
            gongche_display = " ".join(
                f"{g.lyric}[{' '.join(n.gongche for n in g.notes)}]"
                for g in line_groups
            )
            notes_in_line = sum(len(g.notes) for g in line_groups)
            print(f"  L{line_no+1} [{line_text}]: {notes_in_line}音 → {gongche_display[:100]}{'...' if len(gongche_display) > 100 else ''}")
    else:
        print(f"解析到 {len(gc_groups)} 组:")
        for g in gc_groups:
            gcs = [n.gongche for n in g.notes]
            print(f"  {g.lyric}: {gcs} ({len(g.notes)}音)")

    # 工尺谱文本
    txt_path = out_dir / "rich_generation.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"曲牌: {QUPAI}\n")
        f.write(f"歌词: {NEW_LYRICS}\n\n")
        f.write(f"{raw}\n")
        f.write(f"\n--- Prompt 特征摘要 ---\n")
        f.write(user_prompt[:2000])
        f.write(f"\n\nToken: {resp.usage.total_tokens}  |  耗时: {api_time:.1f}s\n")
    print(f"\n工尺谱: {txt_path}")

    # MusicXML (use first line as title for readability)
    short_title = lyric_lines[0] if is_multi else NEW_LYRICS[:10]
    xml_path = out_dir / "rich_generation.musicxml"
    gongche_to_musicxml(
        lyric_groups=gc_groups,
        output_path=str(xml_path),
        title=f"AI生成·{QUPAI}·{short_title}",
        work_title=f"九宫大成·Rich Prompt·{QUPAI}",
        qupai=QUPAI,
    )
    print(f"MusicXML: {xml_path} ({os.path.getsize(str(xml_path))} bytes)")

    # 元数据
    meta = {
        "qupai": QUPAI,
        "lyrics": NEW_LYRICS,
        "raw_response": raw,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
        "api_time": round(api_time, 2),
        "system_prompt_length": len(system_prompt),
        "user_prompt_length": len(user_prompt),
    }
    json.dump(meta, (out_dir / "rich_generation.meta.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
else:
    print("解析失败!")

print(f"\n===== 完成! 总耗时 {time.time() - t_total:.1f}s =====")
print(f"输出目录: {out_dir}/")
