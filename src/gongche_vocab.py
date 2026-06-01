"""
工尺谱词表与编码模块 — 定义工尺字符集、音高编码、格式转换和LLM输出解析。

工尺谱是中國傳統記譜法，用漢字表示音高。音階從低到高：
合 → 四 → 一 → 上 → 尺 → 工 → 凡 → 六 → 五 → 乙 → 仩 → 伬 → 仜
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# 工尺谱词表定义
# ============================================================================

# 14个工尺谱字符，按音高从低到高排列
GONGCHE_CHARS = [
    "合", "四", "一", "上", "尺", "工", "凡",
    "六", "五", "乙", "仩", "伬", "仜"
]

# 工尺 → 音高数值 (上=0.0 为宫音/主音，每步进1.0=1半音)
GONGCHE_TO_PITCH: Dict[str, float] = {
    "合": -5.0, "四": -3.0, "一": -1.0,
    "上": 0.0,  "尺": 2.0,  "工": 4.0,  "凡": 5.0,
    "六": 7.0,  "五": 9.0,  "乙": 11.0,
    "仩": 12.0, "伬": 14.0, "仜": 16.0,
}

# 工尺 → MIDI 音高 (上 = C4 = 60)
GONGCHE_TO_MIDI: Dict[str, int] = {
    gc: int(60 + pitch) for gc, pitch in GONGCHE_TO_PITCH.items()
}

# 工尺 → MusicXML step+octave
PITCH_TO_STEP_OCTAVE: Dict[float, Tuple[str, int]] = {
    -5.0: ("G", 3),
    -3.0: ("A", 3),
    -1.0: ("B", 3),
    0.0:  ("C", 4),
    2.0:  ("D", 4),
    4.0:  ("E", 4),
    5.0:  ("F", 4),
    7.0:  ("G", 4),
    9.0:  ("A", 4),
    11.0: ("B", 4),
    12.0: ("C", 5),
    14.0: ("D", 5),
    16.0: ("E", 5),
}

# 反向映射
PITCH_TO_GONGCHE: Dict[float, str] = {v: k for k, v in GONGCHE_TO_PITCH.items()}
MIDI_TO_GONGCHE: Dict[int, str] = {v: k for k, v in GONGCHE_TO_MIDI.items()}


# ============================================================================
# 时长标记
# ============================================================================

# 文本时长标记 → 实际时长 (四分音符比例)
DURATION_MARKERS = {
    "":    0.25,   # 默认 = 四分音符
    "(O)": 0.5,    # 二分音符
    "(x)": 0.125,  # 八分音符
    "(o)": 0.0625, # 十六分音符
}


def duration_to_marker(dur: float) -> str:
    """时长 → 文本标记"""
    for marker, val in DURATION_MARKERS.items():
        if abs(dur - val) < 0.001:
            return marker
    # 回退：直接显示数字
    return f"({dur:.3f})"


def marker_to_duration(marker: str) -> float:
    """文本标记 → 时长"""
    return DURATION_MARKERS.get(marker, 0.25)


# ============================================================================
# 工尺谱文本格式
# ============================================================================

# 紧凑格式正则：字[工尺1 工尺2 ...(标记)]
# 例如: 風[工] 和[尺 工] 日[上 尺] 陽[尺 工(O)]
GONGCHE_COMPACT_PATTERN = re.compile(
    r'(\S+)\[([^\]]+)\]'
)

# 提取工尺+标记: "尺 工(O)" → [("尺", ""), ("工", "(O)")]
GONGCHE_NOTE_PATTERN = re.compile(
    r'([' + ''.join(GONGCHE_CHARS) + r'])(\([Oxo]\))?'
)


def format_lyric_gongche_pairs(groups: List["LyricNoteGroup"],
                                use_markers: bool = True) -> str:
    """
    将歌词-音符分组列表格式化为紧凑工尺谱文本。

    Args:
        groups: 歌词音符分组列表
        use_markers: 是否使用时长标记

    Returns:
        "風[工] 和[尺 工] 日[上 尺] ..."
    """
    parts = []
    for g in groups:
        notes_str = []
        for i, note in enumerate(g.notes):
            gc = note.gongche
            if use_markers and i == 0 and len(g.notes) == 1:
                marker = duration_to_marker(note.duration)
            elif use_markers and i > 0:
                marker = duration_to_marker(note.duration)
            elif use_markers and abs(note.duration - 0.25) > 0.001:
                marker = duration_to_marker(note.duration)
            else:
                marker = ""
            notes_str.append(f"{gc}{marker}")
        parts.append(f"{g.lyric}[{' '.join(notes_str)}]")
    return " ".join(parts)


def format_gongche_sequence(notes: List["NoteEvent"],
                             use_markers: bool = True) -> str:
    """
    将音符列表格式化为工尺谱序列字符串。

    Returns:
        "工 尺 工 上 尺 工 五 五 六 工 尺 工(O)"
    """
    parts = []
    for n in notes:
        gc = n.gongche
        marker = ""
        if use_markers and abs(n.duration - 0.25) > 0.001:
            marker = duration_to_marker(n.duration)
        parts.append(f"{gc}{marker}")
    return " ".join(parts)


def parse_compact_gongche(text: str) -> List["LyricNoteGroup"]:
    """
    解析紧凑工尺谱文本，返回歌词-音符分组列表。

    输入: "風[工] 和[尺 工] 日[上 尺] 陽[尺 工(O)]"
    输出: List[LyricNoteGroup]

    异常处理：
    - LLM 可能输出格式不规范，需要容错
    """
    from .data_loader import NoteEvent, LyricNoteGroup

    matches = GONGCHE_COMPACT_PATTERN.findall(text)
    if not matches:
        return []

    groups = []
    for lyric_char, gongche_str in matches:
        # 解析工尺序列
        note_matches = re.findall(r'([合四一上尺工'
                                  r'凡六五乙仩伬仜])(\([Oxo]\))?',
                                  gongche_str)
        if not note_matches:
            continue

        notes = []
        total = len(note_matches)
        for i, (gc, marker) in enumerate(note_matches):
            duration = marker_to_duration(marker) if marker else 0.25
            if total > 1 and not marker:
                # Multi-note melisma group without explicit markers:
                # first note stays at 0.25 (quarter), middle notes 0.125 (eighth),
                # last note back to 0.25 (quarter) for a natural sustain
                if i == 0:
                    duration = 0.25
                elif i == total - 1:
                    duration = 0.25
                else:
                    duration = 0.125

            notes.append(NoteEvent(
                lyric=lyric_char,
                lyric_modern_tone=0,
                lyric_is_entering=False,
                lyric_guangyun_tone="",
                gongche=gc,
                gongche_pitch=GONGCHE_TO_PITCH.get(gc, 0.0),
                duration=duration,
                beat_id=0,
                line_id=0,
                lyric_id=0,
                is_melisma=(len(notes) > 0),
                melisma_id=0,
                rhythm="",
            ))

        if notes:
            groups.append(LyricNoteGroup(
                lyric=lyric_char,
                lyric_modern_tone=0,
                lyric_is_entering=False,
                lyric_guangyun_tone="",
                notes=notes,
            ))

    return groups


def parse_llm_response(response_text: str) -> Tuple[List["LyricNoteGroup"], str]:
    """
    解析 LLM 的原始响应，提取工尺谱旋律。

    LLM 可能在响应中包含额外文本（解释、描述等），
    需要从中提取出工尺谱旋律部分。

    Returns:
        (groups, cleaned_gongche_text)
    """
    from .data_loader import NoteEvent, LyricNoteGroup

    text = response_text.strip()

    # 策略1: 查找 "旋律:" 或 "输出旋律:" 后的内容
    melody_patterns = [
        r'(?:输出)?旋律[：:]\s*(.+?)(?:\n|$)',
        r'(?:工尺谱|工尺)[：:]\s*(.+?)(?:\n|$)',
    ]
    for pat in melody_patterns:
        m = re.search(pat, text)
        if m:
            text = m.group(1).strip()
            break

    # 策略2: 直接提取所有 字[工尺...] 模式
    groups = parse_compact_gongche(text)
    if groups:
        gongche_text = format_lyric_gongche_pairs(groups, use_markers=True)
        return groups, gongche_text

    # 策略3: 回退 — 逐行查找类似模式
    for line in text.split("\n"):
        line = line.strip()
        groups = parse_compact_gongche(line)
        if groups:
            gongche_text = format_lyric_gongche_pairs(groups, use_markers=True)
            return groups, gongche_text

    # 策略4: 如果完全没有匹配，尝试清洗后提取
    # 去掉所有中文标点、英文标点，只保留工尺字符和方括号结构
    cleaned = re.sub(r'[^合四一上尺工'
                     r'凡六五乙仩伬仜'
                     r'\[\]\(\)Oxo\s\w]', '', text)
    groups = parse_compact_gongche(cleaned)
    if groups:
        gongche_text = format_lyric_gongche_pairs(groups, use_markers=True)
        return groups, gongche_text

    return [], ""


# ============================================================================
# 音高与旋律工具
# ============================================================================


def gongche_to_midi_pitch(gongche: str) -> int:
    """工尺字符 → MIDI音高"""
    return GONGCHE_TO_MIDI.get(gongche, 60)


def gongche_to_step_octave(gongche: str) -> Tuple[str, int]:
    """工尺字符 → (step, octave) 用于 MusicXML"""
    pitch = GONGCHE_TO_PITCH.get(gongche, 0.0)
    return PITCH_TO_STEP_OCTAVE.get(pitch, ("C", 4))


def get_pitch_range(notes: List[str]) -> Tuple[float, float]:
    """获取工尺序列的音域范围"""
    pitches = [GONGCHE_TO_PITCH.get(gc, 0.0) for gc in notes if gc in GONGCHE_TO_PITCH]
    if not pitches:
        return (0.0, 0.0)
    return (min(pitches), max(pitches))


def get_pitch_intervals(notes: List[str]) -> List[float]:
    """获取音程序列（相邻音符之间的音高差）"""
    pitches = [GONGCHE_TO_PITCH.get(gc, 0.0) for gc in notes if gc in GONGCHE_TO_PITCH]
    if len(pitches) < 2:
        return []
    return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]


def is_valid_gongche_char(char: str) -> bool:
    """检查是否为有效工尺字符"""
    return char in GONGCHE_TO_PITCH


def melody_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个工尺旋律的相似度 (基于编辑距离)。

    返回值: 0.0 (完全不相似) ~ 1.0 (完全相同)
    """
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    # 编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 删除
                dp[i][j-1] + 1,      # 插入
                dp[i-1][j-1] + cost, # 替换
            )

    edit_dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (edit_dist / max_len)


# ============================================================================
# 统计工具
# ============================================================================


def get_pitch_distribution(notes: List[str]) -> Dict[str, float]:
    """获取工尺序列的音高分布（归一化）"""
    from collections import Counter
    if not notes:
        return {}
    counts = Counter(notes)
    total = len(notes)
    return {gc: count / total for gc, count in counts.items()}


def get_interval_distribution(notes: List[str]) -> Dict[float, float]:
    """获取音程分布"""
    from collections import Counter
    intervals = get_pitch_intervals(notes)
    if not intervals:
        return {}
    counts = Counter(intervals)
    total = len(intervals)
    return {k: v / total for k, v in counts.items()}
