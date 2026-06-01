"""
Rich Prompt 构建器 — 将特征提取结果整合进 System Prompt 和 Few-Shot 样例。

特征来源: build_features.py → features_cache.json/pkl
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from .gongche_vocab import (
    GONGCHE_TO_PITCH, GONGCHE_CHARS, PITCH_TO_GONGCHE,
    format_lyric_gongche_pairs,
)


class RichPromptBuilder:
    """
    基于数据集特征的 Rich Prompt 构建器。

    Usage:
        with open("features_cache.json") as f:
            features = json.load(f)
        builder = RichPromptBuilder(features)
        system, user = builder.build("奉時春", "春風拂柳燕歸來")
    """

    def __init__(self, features: Dict):
        self.features = features
        self.qupai_features = features.get("qupai_features", {})
        self.tone_features = features.get("tone_features", {})
        self.mode_features = features.get("mode_features", {})
        self.region_features = features.get("region_features", {})
        self.global_stats = features.get("global_stats", {})

    # ==================================================================
    # System Prompt
    # ==================================================================

    def build_system_prompt(self, region: str = None, mode: str = None) -> str:
        """构建带有数据特征的 System Prompt"""
        parts = []

        # 角色
        parts.append(
            "你是一位精通《九宫大成南北词宫谱》的中国古典音乐AI作曲大师。"
            "你深刻理解曲牌音乐的结构规律，能根据给定的曲牌特征和新歌词，"
            "创作出风格地道的工尺谱旋律。"
        )

        # 工尺基础
        parts.append(self._gongche_basics())

        # 声调-旋律规律（来自数据）
        parts.append(self._tone_melody_rules())

        # 南词/北词风格差异
        parts.append(self._region_style_rules())

        # 作曲规则
        parts.append(self._composition_rules())

        # 输出格式
        parts.append(self._output_format())

        return "\n\n".join(parts)

    def _gongche_basics(self) -> str:
        """工尺谱基础知识"""
        gs = self.global_stats
        pitch_r = gs.get("pitch_range", {})
        return f"""## 工尺谱基础

工尺谱用汉字表示音高，14个常用字符从低到高：
```
合(-5) → 四(-3) → 一(-1) → 上(0=宫音/C4) → 尺(2) → 工(4) → 凡(5) → 六(7) → 五(9) → 乙(11) → 仩(12) → 伬(14) → 仜(16)
```
括号内为相对于宫音"上"的半音数。

数据集整体音域: {pitch_r.get('min', -5)} ~ {pitch_r.get('max', 16)} 半音
最常用五个音: 上(17.1%) 尺(17.6%) 工(18.8%) 四(13.1%) 六(11.6%)

## 时长标记

每个曲牌有自己的节奏密度。通用标记规则：
- **无标记** = 四分音符（默认，最常用）
- **(O)** = 二分音符（长音，多用于句尾收束）
- **(x)** = 八分音符（短音，用于拖腔的后续快速音）

具体使用频率请参考目标曲牌的统计数据——低密度曲牌（如煞尾）以四分音符为主，
高密度曲牌（如小桃红）八分音符和拖腔更频繁。"""

    def _tone_melody_rules(self) -> str:
        """声调与旋律的对应规律"""
        tf = self.tone_features
        gt = tf.get("guangyun_tones", {})
        mt = tf.get("modern_tones", {})

        rules = ["## 依字行腔 — 声调与旋律的对应规律", ""]

        # 广韵声调
        if gt:
            rules.append("### 基于广韵四声的统计数据\n")
            for tone in ["平", "上", "去", "入"]:
                if tone not in gt:
                    continue
                t = gt[tone]
                top3 = list(t.get("pitch_distribution", {}).keys())[:3]
                top_first = list(t.get("common_first_pitch", {}).keys())[:3]
                melisma = t.get("melisma_rate", 0)
                dur = t.get("avg_duration", 0.25)

                hints = []
                if tone == "平":
                    hints.append("旋律平稳，多用上、尺、工等核心音")
                    hints.append(f"拖腔率{melisma:.0%}，常配较长的旋律线")
                elif tone == "上":
                    hints.append("旋律常从较低音开始上行（如上→尺、四→上）")
                    hints.append(f"拖腔率{melisma:.0%}，较平声略少")
                elif tone == "去":
                    hints.append("旋律多从高到低下行（如六→工→尺→上）")
                    hints.append(f"拖腔率{melisma:.0%}，倾向利落收束")
                elif tone == "入":
                    hints.append("短促有力，多用单音符")
                    hints.append(f"拖腔率{melisma:.0%}（明显低于其他声调）")
                    hints.append(f"平均时长{dur:.3f}（短于其他声调）")

                rules.append(
                    f"**{tone}声**（占总音符{100*t.get('count',0)/696216:.0f}%）："
                    f"常用音 {top3}，起音偏好 {top_first}。{'；'.join(hints)}"
                )

        return "\n".join(rules)

    def _region_style_rules(self) -> str:
        """南词 vs 北词风格差异"""
        rf = self.region_features
        if "南詞" not in rf or "北詞" not in rf:
            return ""

        nan = rf["南詞"]
        bei = rf["北詞"]

        return f"""## 南词 vs 北词风格差异

| 维度 | 南词 ({nan['song_count']}首) | 北词 ({bei['song_count']}首) |
|------|------|------|
| 拖腔率 | {nan['melisma_rate']:.1%} | {bei['melisma_rate']:.1%} |
| 平均密度 | {nan['avg_density']:.2f}音/字 | {bei['avg_density']:.2f}音/字 |
| 音域 | {nan['pitch_range']['max']-nan['pitch_range']['min']:.0f}半音 | {bei['pitch_range']['max']-bei['pitch_range']['min']:.0f}半音 |

**南词特点**: 拖腔更多，旋律更婉转细腻，装饰音丰富
**北词特点**: 拖腔较少，旋律更直接刚健，跳进比例更高

创作时请根据曲牌所属的南词/北词调整风格。"""

    def _composition_rules(self) -> str:
        return """## 作曲要诀

1. **遵循曲牌规律**: 每个曲牌有自己独特的拖腔密度和分布模式。
   部分曲牌一字一音为主（如煞尾密度1.51），另一部分全行华彩（如小桃红密度2.19）。
   请严格参考目标曲牌的统计数据来把控旋律的繁简程度。
2. **起音收音**: 同一曲牌的起音（首字首音）和收音（末字末音）高度一致，
   请严格参考样例中的起收音位置
3. **节奏**: 默认用四分音符。二分(O)常用于句尾末字收束。
   八分(x)用于入声字或高密度曲牌中快速经过的拖腔音。
4. **五声音阶为主**: 多用上-尺-工-六-五（对应 do-re-mi-sol-la），
   四、合作为低音支持，凡、乙作为色彩音少量使用
5. **音域控制**: 旋律音域一般在8-12个半音以内（一个八度左右）"""

    def _qupai_rhythm_profile(self, qupai: str) -> str:
        """根据曲牌的实际拖腔位置分布生成节奏指导"""
        qf = self.qupai_features.get(qupai, {})
        mpos = qf.get("melisma_by_position", {})
        density = qf.get("density", {}).get("avg_notes_per_lyric", 0)
        melisma_rate = qf.get("density", {}).get("avg_melisma_ratio", 0)

        if not mpos or not density:
            return ""

        pos_labels = ["句首", "句前", "句中", "句后", "句尾"]
        parts = [f"\n## 曲牌《{qupai}》的节奏特征\n"]

        level = "高" if melisma_rate > 0.35 else ("中" if melisma_rate > 0.15 else "低")
        parts.append(f"该曲牌整体装饰度: **{level}** (密度{density:.1f}音/字, 拖腔率{melisma_rate*100:.0f}%)")

        # Positional pattern
        if len(mpos) >= 3:
            parts.append(f"\n拖腔在句中的分布规律（百分比=该位置的字出现拖腔的概率）:")
            for ki in sorted(mpos.keys(), key=int):
                bi = int(ki)
                v = mpos[ki]
                bar = "█" * int(v * 30)
                parts.append(f"  {pos_labels[bi]:4s} ({bi*20}-{(bi+1)*20}%位置): {v*100:3.0f}% {bar}")

            # Generate guidance
            if len(mpos) >= 5:
                vals = [mpos[str(i)] for i in range(5)]
                start_lo = vals[0] < 0.3
                end_hi = vals[4] > 0.7
                if start_lo and end_hi:
                    parts.append(f"\n→ 此曲牌拖腔集中在句尾，句中保持简洁。每行仅句末字拖腔。")
                elif all(v > 0.5 for v in vals):
                    parts.append(f"\n→ 此曲牌全行华彩，每字都可能配多个音。旋律应丰满流畅。")
                else:
                    parts.append(f"\n→ 此曲牌拖腔分布均匀，按参考样例的节奏模式创作。")
        else:
            parts.append(f"\n参考样例中的音符密度和拖腔位置来创作。")

        return "\n".join(parts)

    def _output_format(self) -> str:
        return """## 输出格式

每行歌词单独输出一行旋律。
格式: `<字>[<工尺> <工尺>...] <字>[<工尺>...] ...`
时长标记: 无=四分 (O)=二分 (x)=八分

根据曲牌装饰度决定拖腔密度：低装饰度曲牌大部分字单音，高装饰度曲牌每字都可拖腔。
请严格参考目标曲牌统计数据中的拖腔位置分布来安排。

每行歌词对应一行"旋律: ..."。
只输出旋律，不要任何解释。"""

    # ==================================================================
    # 曲牌特征摘要
    # ==================================================================

    def build_qupai_feature_summary(self, qupai: str) -> str:
        """为指定曲牌构建特征摘要（嵌入 Few-Shot 之前）"""
        qf = self.qupai_features.get(qupai)
        if not qf:
            return ""

        parts = [f"\n## 曲牌《{qupai}》数据特征\n"]

        # 基本统计
        parts.append(f"- 数据集中出现 **{qf['song_count']}** 次")

        # 起音收音
        starts = list(qf.get("start_pitches", {}).items())
        if starts:
            starts_sorted = sorted(starts, key=lambda x: -x[1])
            parts.append(f"- 起音（首字首音）偏好: {', '.join(f'{k}({v}次)' for k,v in starts_sorted[:3])}")

        ends = list(qf.get("end_pitches", {}).items())
        if ends:
            ends_sorted = sorted(ends, key=lambda x: -x[1])
            parts.append(f"- 收音（末字末音）偏好: {', '.join(f'{k}({v}次)' for k,v in ends_sorted[:3])}")

        # 音高
        ps = qf.get("pitch_stats", {})
        parts.append(f"- 音高范围: {ps.get('min',0):.0f} ~ {ps.get('max',0):.0f}，均值{ps.get('mean',0):.1f}±{ps.get('std',0):.1f}")

        # 密度
        ds = qf.get("density", {})
        parts.append(f"- 每字音符数: 平均{ds.get('avg_notes_per_lyric',0):.2f}，拖腔率{ds.get('avg_melisma_ratio',0):.1%}")

        # 音程偏好
        iv = qf.get("interval_distribution", {})
        if iv:
            # 找出最常见的音程
            top_iv = sorted(iv.items(), key=lambda x: -x[1])[:4]
            iv_desc = []
            for k, v in top_iv:
                kf = float(k)
                if abs(kf) < 0.5:
                    s = "同音"
                elif kf > 0:
                    s = f"上行{kf:.0f}半音"
                else:
                    s = f"下行{-kf:.0f}半音"
                iv_desc.append(f"{s}({v}次)")
            parts.append(f"- 常见音程: {', '.join(iv_desc)}")

        # 宫调
        modes = qf.get("mode_distribution", {})
        if modes:
            parts.append(f"- 所属宫调: {', '.join(list(modes.keys())[:3])}")

        # 南/北
        regions = qf.get("region_distribution", {})
        if regions:
            parts.append(f"- 板块: {', '.join(f'{k}({v})' for k,v in list(regions.items())[:2])}")

        return "\n".join(parts)

    # ==================================================================
    # 宫调特征
    # ==================================================================

    def build_mode_feature_summary(self, mode: str) -> str:
        """构建宫调特征摘要"""
        mf = self.mode_features.get(mode)
        if not mf:
            return ""

        parts = [f"\n### 宫调「{mode}」特征\n"]
        parts.append(f"- 包含 {mf['song_count']} 首歌曲，{mf['qupai_count']} 种曲牌")

        pr = mf.get("pitch_range", {})
        parts.append(f"- 音域: {pr.get('min',0):.0f} ~ {pr.get('max',0):.0f}，中位数{pr.get('median',0):.0f}")

        top = list(mf.get("top_pitches", {}).items())[:5]
        parts.append(f"- 核心音高: {', '.join(f'{k}({v}次)' for k,v in top)}")

        # 节奏
        parts.append(f"- 散板率: {mf.get('rubato_rate',0):.1%}")
        parts.append(f"- 装饰度(依样): {mf.get('avg_tone_yiyang',0):.2f}")

        # 情感
        em = mf.get("emotional_hint", "")
        if em:
            parts.append(f"- 风格提示: {em}")

        return "\n".join(parts)

    # ==================================================================
    # 完整构建
    # ==================================================================

    def build(
        self,
        qupai: str,
        lyrics: str,
        examples: Optional[List[Dict]] = None,
        region: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        构建完整的 System + User prompt（含全部特征）。

        Returns:
            (system_prompt, user_prompt)
        """
        # 推断 region/mode
        qf = self.qupai_features.get(qupai, {})
        if not region and qf:
            regions = qf.get("region_distribution", {})
            region = max(regions, key=regions.get) if regions else None
        if not mode and qf:
            modes = qf.get("mode_distribution", {})
            mode = max(modes, key=modes.get) if modes else None

        # System Prompt
        system = self.build_system_prompt(region=region, mode=mode)

        # 宫调特征补充
        if mode and mode in self.mode_features:
            system += "\n\n" + self.build_mode_feature_summary(mode)

        # User Prompt
        user_parts = []

        # 曲牌特征
        qupai_summary = self.build_qupai_feature_summary(qupai)
        if qupai_summary:
            user_parts.append(qupai_summary)

        # Few-Shot 样例
        if examples:
            user_parts.append(self._format_rich_examples(examples, qupai))

        # 生成任务
        user_parts.append(self._build_task(qupai, lyrics, region, mode))

        return system, "\n\n".join(user_parts)

    def _format_rich_examples(
        self, examples: List[Dict], qupai: str, max_n: int = 2
    ) -> str:
        """格式化 Few-Shot 样例（含统计标注，支持多行）"""
        parts = ["\n## 参考样例\n"]

        for i, ex in enumerate(examples[:max_n]):
            lyrics = ex.get("lyrics", "")
            lyric_note_groups = ex.get("lyric_note_groups", [])

            total_notes = sum(len(g.notes) for g in lyric_note_groups) if lyric_note_groups else ex.get("note_count", 0)
            n_lyrics = len(lyric_note_groups) if lyric_note_groups else len(lyrics)
            density = total_notes / max(n_lyrics, 1)

            parts.append(f"### 样例 {i+1}")
            if ex.get("polyu_id"):
                parts.append(f"ID: {ex['polyu_id']}  |  {n_lyrics}字 {total_notes}音 密度{density:.1f}")

            # 按 line_id 分组输出（多行歌词逐行显示）
            lines = defaultdict(list)
            for g in lyric_note_groups:
                line_id = g.notes[0].line_id if g.notes else 0
                lines[line_id].append(g)

            if len(lines) > 1:
                # 多行：逐行输出
                for lid in sorted(lines.keys()):
                    line_groups = lines[lid]
                    line_text = "".join(str(g.lyric) for g in line_groups)
                    line_melody = format_lyric_gongche_pairs(line_groups, use_markers=True)
                    parts.append(f"  {line_text}")
                    parts.append(f"  旋律: {line_melody}")
            else:
                # 单行
                lyrics_text = "".join(str(g.lyric) for g in lyric_note_groups)
                gongche_text = format_lyric_gongche_pairs(lyric_note_groups, use_markers=True)
                parts.append(f"歌词: {lyrics_text}")
                parts.append(f"旋律: {gongche_text}")

        return "\n".join(parts)

    def _build_task(
        self, qupai: str, lyrics: str,
        region: Optional[str], mode: Optional[str]
    ) -> str:
        """构建当前生成任务"""
        n_chars = len(lyrics)
        lyrics_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        n_lines = len(lyrics_lines)
        is_multi_line = n_lines > 1

        # 使用曲牌自身密度（不做人工上限）
        qf = self.qupai_features.get(qupai, {})
        qf_density = qf.get("density", {}).get("avg_notes_per_lyric", 0)
        if not qf_density or np.isnan(qf_density):
            qf_density = 1.78
        target_density = qf_density

        expected_notes = int(n_chars * target_density)
        melisma_rate = qf.get("density", {}).get("avg_melisma_ratio", 0)

        parts = [f"\n## 生成任务\n"]
        parts.append(f"曲牌: 《{qupai}》")
        if region:
            parts.append(f"板块: {region}")
        if mode:
            parts.append(f"宫调: {mode}")

        # 曲牌自身节奏特征
        rhythm_profile = self._qupai_rhythm_profile(qupai)

        if is_multi_line:
            parts.append(f"新歌词({n_chars}字, {n_lines}行):")
            for line in lyrics_lines:
                parts.append(f"  {line}")
            parts.append(f"\n请参考上述样例和曲牌特征，为以上{n_lines}行歌词逐行生成工尺谱旋律。")
            parts.append(f"每行歌词单独输出一行'旋律: ...'。")
            parts.append(f"此曲牌密度 {target_density:.1f} 音/字（拖腔率{melisma_rate*100:.0f}%），预期总音符约{expected_notes}个。")
            if rhythm_profile:
                parts.append(rhythm_profile)
        else:
            parts.append(f"新歌词({n_chars}字): {lyrics}")
            parts.append(f"\n请参考上述样例和曲牌特征，创作为以上歌词的工尺谱旋律。")
            parts.append(f"此曲牌密度 {target_density:.1f} 音/字（拖腔率{melisma_rate*100:.0f}%），预期总音符约{expected_notes}个。")
            if rhythm_profile:
                parts.append(rhythm_profile)

        return "\n".join(parts)



