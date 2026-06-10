"""
定量评估模块 — 评估生成旋律与数据集的风格相似度和声调对齐度。

评估维度：
  风格相似度（70%权重）:
    1. format_compliance   (0.15) — 工尺字符合格率
    2. pitch_distribution  (0.25) — 音高分布 JS 散度 vs 曲牌参考
    3. interval_distribution (0.15) — 音程分布余弦相似度
    4. density_match       (0.15) — 每字音符数 vs 曲牌参考
    5. melisma_match       (0.10) — 拖腔率及位置分布
    6. boundary_match      (0.10) — 起音/收音匹配
    7. range_match         (0.10) — 音域跨度匹配

  声调对齐度（30%权重）:
    - 平声: 音高稳定性、核心音偏好
    - 上声: 上行倾向、起始音偏低
    - 去声: 下行倾向
    - 入声: 时值短、少拖腔
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np

from .gongche_vocab import (
    GONGCHE_TO_PITCH,
    get_pitch_distribution,
    get_pitch_range,
    get_pitch_intervals,
)
from .data_loader import LyricNoteGroup

logger = logging.getLogger(__name__)


# ============================================================================
# 结果数据结构
# ============================================================================

@dataclass
class QuantitativeResult:
    """定量评估完整结果"""
    piece_id: str
    qupai: str
    lyrics: str

    # 风格相似度
    style_scores: Dict[str, float] = field(default_factory=dict)
    style_overall: float = 0.0

    # 声调对齐度 — 规则判断
    tone_rule_scores: Dict[str, float] = field(default_factory=dict)
    tone_rule_overall: float = 0.0

    # 声调对齐度 — 数据驱动（与 tone_features 统计对比）
    tone_data_scores: Dict[str, float] = field(default_factory=dict)
    tone_data_overall: float = 0.0

    # 声调综合（规则 + 数据 各50%）
    tone_scores: Dict[str, float] = field(default_factory=dict)
    tone_overall: float = 0.0

    # 综合
    overall_score: float = 0.0

    # 详细数据（生成值 vs 参考值对比）
    metric_details: Dict[str, Any] = field(default_factory=dict)

    # 逐字声调分析
    char_details: List[Dict] = field(default_factory=list)

    # 时间戳
    timestamp: str = ""


# ============================================================================
# 主评估器
# ============================================================================

class QuantitativeEvaluator:
    """
    定量评估器 — 使用 features_cache 作为参考数据集。

    Usage:
        import json
        with open("features_cache.json", "r", encoding="utf-8") as f:
            features = json.load(f)
        evaluator = QuantitativeEvaluator(features)
        result = evaluator.evaluate(gc_groups, lyrics, qupai)
        print(f"综合评分: {result.overall_score:.0%}")
    """

    # 风格相似度权重（LLM / 曲牌模式）
    STYLE_WEIGHTS = {
        "format_compliance": 0.15,
        "pitch_distribution": 0.25,
        "interval_distribution": 0.15,
        "density_match": 0.15,
        "melisma_match": 0.10,
        "boundary_match": 0.10,
        "range_match": 0.10,
    }

    # 风格相似度权重（Transformer / 宫调模式，仅 3 个有效指标）
    MODE_STYLE_WEIGHTS = {
        "pitch_distribution": 0.45,
        "interval_distribution": 0.30,
        "range_match": 0.25,
    }

    # 声调在数据集中的典型分布（用于加权平均）
    TONE_POPULATION_WEIGHTS = {
        "平": 0.407,
        "上": 0.177,
        "去": 0.303,
        "入": 0.112,
    }

    # 所有评估指标的计算说明（用于报告输出）
    METRIC_DESCRIPTIONS = {
        "format_compliance": {
            "label": "格式合规",
            "method": "合法工尺字组数 / 总字组数；若解析完全失败但 LLM 返回了文本，给予 20% 部分信用",
            "reference": "GONGCHE_TO_PITCH（14 字符工尺映射字典）",
            "meaning": "衡量 LLM 输出是否能被解析为合法的工尺谱格式。0=完全不可解析, 1=全部合法",
        },
        "pitch_distribution": {
            "label": "音高分布匹配",
            "method": "Jensen-Shannon 散度 → 相似度 = 1/(1+√JS)。70% 权重与该曲牌数据集音高分布对比，30% 权重与全数据集音高分布对比",
            "reference": "features_cache.json → qupai_features[曲牌].pitch_distribution（该曲牌各工尺字使用频次）+ global_stats.overall_pitch_distribution（全数据集音高分布）",
            "meaning": "衡量生成旋律的音高使用频率是否与该曲牌/全数据集的工尺偏好一致。0=完全不同, 1=完全一致",
        },
        "interval_distribution": {
            "label": "音程分布匹配",
            "method": "生成旋律的相邻音高差分为 7 类（同音/级进上/级进下/跳进上/跳进下/大跳上/大跳下），与参考分布做余弦相似度(60%) + 级进比例差分(40%)",
            "reference": "features_cache.json → qupai_features[曲牌].interval_distribution（该曲牌中音程的分布直方图）",
            "meaning": "衡量生成旋律的音程跳跃模式（级进 vs 跳进 vs 大跳）是否与数据集同名曲牌一致。0=完全不同, 1=完全一致",
        },
        "density_match": {
            "label": "密度匹配",
            "method": "min(生成密度, 参考密度) / max(生成密度, 参考密度)；密度 = 总音符数 / 总字数。无参考时使用全数据集均值 1.7",
            "reference": "features_cache.json → qupai_features[曲牌].density.avg_notes_per_lyric（该曲牌每字平均配音符数）",
            "meaning": "衡量生成的每字音符密度是否与数据集同名曲牌一致。0=严重偏离, 1=完全吻合",
        },
        "melisma_match": {
            "label": "拖腔匹配",
            "method": "(1) 拖腔率偏差(容忍带 ±0.08)：1−|gen_rate−ref_rate|/ref_rate, 50%权重。(2) 5 位置 bin(Spearman 秩相关)：看拖腔在句首/句前/句中/句后/句尾的趋势是否一致, 50%权重",
            "reference": "features_cache.json → qupai_features[曲牌].density.avg_melisma_ratio（平均拖腔率）+ melisma_by_position（各句中位置拖腔概率）",
            "meaning": "衡量生成的拖腔密度和句中位置分布是否与数据集同名曲牌一致。0=完全不同, 1=完全一致",
        },
        "boundary_match": {
            "label": "起收音匹配",
            "method": "首字首音(50%) + 末字末音(50%)。精确匹配该曲牌常用起/收音得满分，±3 半音内邻近音按距离递减给分：1/(1+|pitch_diff|)。若曲牌样本量 <3 首，自动融合同名曲牌双向子串匹配",
            "reference": "features_cache.json → qupai_features[曲牌].start_pitches + end_pitches（该曲牌最常用的首音和末音工尺字及频次）",
            "meaning": "衡量生成的起音和收音是否使用了数据集同名曲牌的惯用工尺。0=不匹配, 1=精确匹配",
        },
        "range_match": {
            "label": "音域匹配",
            "method": "min(生成音域跨度, 参考跨度) / max(生成音域跨度, 参考跨度)；无参考时, 跨度 3~14 半音得 0.8 分",
            "reference": "features_cache.json → qupai_features[曲牌].pitch_stats.min/max（该曲牌在数据集中的最低和最高音高）",
            "meaning": "衡量生成的音域（最高音-最低音）是否在数据集同名曲牌的合理范围内。0=严重偏离, 1=完全吻合",
        },
        "tone_rule_scores": {
            "label": "声调对齐—规则判断",
            "method": "基于传统'依字行腔'规则逐字打分：平声看音高稳定性(方差) + 核心音匹配；上声看上/升倾向(单音符由跨字音程补偿) + 低位首音(百分位)；去声看下/降倾向 + 高位首音；入声看短时值(≤2倍参考时长) + 单音偏好。四子项(stability/contour/brevity/first_pitch)等权平均",
            "reference": "tone_features.guangyun_tones[声调].common_first_pitch（该声调在全数据集中最常用的首音）+ avg_duration（该声调在全数据集中的平均时值）",
            "meaning": "衡量生成旋律是否符合传统曲唱中'依字行腔'的声调规则。0=完全不遵循, 1=完美遵循",
        },
        "tone_data_scores": {
            "label": "声调对齐—数据驱动",
            "method": "按声调聚合后与全数据集该声调的统计特征对比：(1) 首音工尺分布与数据集该声调音高分布的 JS 散度 (2) 首音在该声调常用首音中的匹配率 (3) 拖腔率匹配 (4) 平均时值匹配。四项等权平均",
            "reference": "tone_features.guangyun_tones[声调].pitch_distribution（全数据集 69 万音符中该声调的音高分布）/ common_first_pitch / melisma_rate / avg_duration",
            "meaning": "衡量生成旋律的声调—音高对应关系是否与全数据集的统计模式一致。0=完全不同, 1=完全一致",
        },
        "style_overall": {
            "label": "风格相似度(综合)",
            "method": "7 个子指标加权求和：格式合规 0.15 + 音高分布 0.25 + 音程分布 0.15 + 密度 0.15 + 拖腔 0.10 + 起收音 0.10 + 音域 0.10",
            "reference": "features_cache.json 中该曲牌的全部统计特征（qupai_features + global_stats）",
            "meaning": "生成旋律与数据集同名曲牌在 7 个维度上的整体风格相似程度。0=完全不像, 1=完全一致",
        },
        "tone_overall": {
            "label": "声调对齐度(综合)",
            "method": "规则判断(50%) + 数据驱动(50%)。各声调按数据集分布加权（平 40.7% / 上 17.7% / 去 30.3% / 入 11.2%）",
            "reference": "tone_features.guangyun_tones 中平上去入四声的全部统计特征",
            "meaning": "生成旋律在声调—旋律关系上的综合质量。0=完全失调, 1=完美对齐",
        },
        "overall_score": {
            "label": "综合评分",
            "method": "0.70 × 风格相似度 + 0.30 × 声调对齐度",
            "reference": "上述所有参考来源的综合",
            "meaning": "生成旋律的整体质量评估，风格为主要维度(70%)，声调为辅助维度(30%)",
        },
    }

    # 宫调模式指标说明（Transformer 模型使用 mode_features 作为参考）
    MODE_METRIC_DESCRIPTIONS = {
        "pitch_distribution": {
            "label": "音高分布匹配",
            "method": "Jensen-Shannon 散度 → 相似度 = 1/(1+√JS)。生成首音工尺分布 vs 该宫调在数据集中的音高分布",
            "reference": "features_cache.json → mode_features[宫调].pitch_distribution（该宫调下所有曲目的工尺字频次归一化分布）",
            "meaning": "衡量生成旋律的音高使用频率是否与该宫调的整体统计一致。0=完全不同, 1=完全一致",
        },
        "interval_distribution": {
            "label": "音程分布匹配",
            "method": "生成相邻音高差 7 分类（同音/级进/跳进/大跳）与宫调参考分布做余弦相似度(60%) + 级进比例差分(40%)",
            "reference": "features_cache.json → mode_features[宫调].interval_distribution（该宫调已分类的 step_up/down/leap_up/down/big_up/down/same_pitch）",
            "meaning": "衡量生成旋律的音程跳跃模式是否与该宫调的整体统计一致。0=完全不同, 1=完全一致",
        },
        "range_match": {
            "label": "音域匹配",
            "method": "min(生成音域跨度, 参考跨度) / max(生成音域跨度, 参考跨度)",
            "reference": "features_cache.json → mode_features[宫调].pitch_range.min/max（该宫调在数据集中的最低和最高音高）",
            "meaning": "衡量生成的音域是否在该宫调的合理范围内。0=严重偏离, 1=完全吻合",
        },
        "style_overall": {
            "label": "风格相似度(宫调)",
            "method": "3 个可用指标加权求和：音高分布 0.45 + 音程分布 0.30 + 音域 0.25。注：宫调级特征不含起收音/拖腔/密度数据，故不可用",
            "reference": "features_cache.json → mode_features[宫调]（该宫调在 6563 首曲目中聚合统计）",
            "meaning": "生成旋律与该宫调在音高使用、音程跳跃、音域三个维度上的整体相似程度",
        },
    }

    def __init__(self, features: Dict):
        """
        Args:
            features: features_cache.json 加载后的 dict
        """
        self.features = features
        self.qupai_features = features.get("qupai_features", {})
        self.tone_features = features.get("tone_features", {})
        self.global_stats = features.get("global_stats", {})

        # 延迟加载声调标注器
        self._tone_aligner = None

        # 每次评估独立的指标存储
        self._metric_store: Dict[str, Any] = {}

    @property
    def tone_aligner(self):
        if self._tone_aligner is None:
            from .tone_aligner import ToneAligner
            self._tone_aligner = ToneAligner()
        return self._tone_aligner

    def evaluate(
        self,
        generated_groups: List[LyricNoteGroup],
        lyrics: str,
        qupai: str,
        raw_response: str = "",
        piece_id: str = "",
        reference_mode: str = "llm",
    ) -> QuantitativeResult:
        """
        评估生成的旋律。

        Args:
            generated_groups: parse_compact_gongche() 返回的 LyricNoteGroup 列表
            lyrics: 原始输入歌词
            qupai: 目标曲牌名（或宫调名，当 reference_mode="transformer" 时）
            raw_response: LLM 原始响应（用于格式合规检测）
            piece_id: 作品标识
            reference_mode: "llm"（曲牌评估）或 "transformer"（宫调评估）

        Returns:
            QuantitativeResult
        """
        from datetime import datetime

        if not piece_id:
            # 清洗曲牌名和歌词前6字作为 ID
            safe_qp = re.sub(r'[《》\s]', '', qupai)
            safe_ly = lyrics.replace('\n', '')[:6]
            piece_id = f"{safe_qp}_{safe_ly}"

        # 每次评估前清空指标存储（防止 singleton 复用导致残留）
        self._metric_store.clear()

        result = QuantitativeResult(
            piece_id=piece_id,
            qupai=qupai,
            lyrics=lyrics,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # ---- 风格相似度评估 ----
        if reference_mode == "transformer":
            result.style_scores = self._evaluate_style_mode(
                generated_groups, qupai
            )
            result.style_overall = self._weighted_sum(
                result.style_scores, self.MODE_STYLE_WEIGHTS
            )
        else:
            result.style_scores = self._evaluate_style(
                generated_groups, qupai, raw_response
            )
            result.style_overall = self._weighted_sum(
                result.style_scores, self.STYLE_WEIGHTS
            )

        # ---- 声调对齐度评估（规则 + 数据驱动）----
        tone_result = self._evaluate_tone_alignment(
            generated_groups, lyrics, qupai
        )
        result.tone_rule_scores = tone_result["rule_scores"]
        result.tone_rule_overall = tone_result["rule_overall"]
        result.tone_data_scores = tone_result["data_scores"]
        result.tone_data_overall = tone_result["data_overall"]
        result.char_details = tone_result["char_details"]
        result.metric_details.update(tone_result.get("details", {}))
        # 合并风格指标的 gen/ref 对比数据
        result.metric_details.update(self._metric_store)

        # 声调综合：规则 50% + 数据驱动 50%
        combined_tone = {}
        for tone in ["平", "上", "去", "入"]:
            r = result.tone_rule_scores.get(tone)
            d = result.tone_data_scores.get(tone)
            if r is not None and d is not None:
                combined_tone[tone] = round(0.5 * r + 0.5 * d, 4)
            elif r is not None:
                combined_tone[tone] = r
            elif d is not None:
                combined_tone[tone] = d
            else:
                combined_tone[tone] = None
        result.tone_scores = combined_tone

        # 综合 tone_overall = 0.5*rule + 0.5*data
        result.tone_overall = round(
            0.5 * result.tone_rule_overall + 0.5 * result.tone_data_overall, 4
        )

        # ---- 综合评分 ----
        result.overall_score = 0.70 * result.style_overall + 0.30 * result.tone_overall
        result.overall_score = round(result.overall_score, 4)
        result.style_overall = round(result.style_overall, 4)
        result.tone_overall = round(result.tone_overall, 4)

        return result

    # ========================================================================
    # 风格相似度
    # ========================================================================

    def _evaluate_style_mode(
        self,
        groups: List[LyricNoteGroup],
        gongdiao: str,
    ) -> Dict[str, float]:
        """宫调级风格评估（Transformer 模型专用，使用 mode_features）。

        mode_features 的数据结构不同于 qupai_features：
        - pitch_distribution: 已归一化 ({gc: prob})
        - interval_distribution: 已分类 ({step_up: prob, ...})
        - pitch_range: {min, max, mean, std}
        - 不含: start_pitches, end_pitches, melisma, density
        """
        # 查找宫调 profile
        mode_name = gongdiao.strip()
        mode_prof = self.features.get("mode_features", {}).get(mode_name)
        if not mode_prof:
            # 尝试模糊匹配
            for key in self.features.get("mode_features", {}):
                if mode_name in key or key in mode_name:
                    mode_prof = self.features["mode_features"][key]
                    break
        if not mode_prof:
            # 退化为全局统计
            logger.warning(f"宫调「{gongdiao}」不在 mode_features 中，退化为全局统计")
            mode_prof = {}

        scores = {}

        # ---- 音高分布匹配 ----
        gen_notes = []
        for g in groups:
            gen_notes.extend(n.gongche for n in g.notes if n.gongche in GONGCHE_TO_PITCH)
        gen_dist = get_pitch_distribution(gen_notes)
        if gen_dist and mode_prof:
            ref_dist = mode_prof.get("pitch_distribution", {})
            if ref_dist:
                scores["pitch_distribution"] = self._js_divergence_score(gen_dist, ref_dist)
            else:
                scores["pitch_distribution"] = 0.5
        elif gen_dist:
            scores["pitch_distribution"] = 0.5
        else:
            scores["pitch_distribution"] = 0.0

        # ---- 音程分布匹配 ----
        gen_intervals = get_pitch_intervals(gen_notes)
        if gen_intervals:
            gen_cats = self._categorize_intervals(gen_intervals)
            ref_cats = mode_prof.get("interval_distribution", {}) if mode_prof else {}
            if ref_cats:
                # ref_cats 通常是归一化的，字段名可能略有差异
                # 标准化字段名
                actual_keys = set(gen_cats.keys())
                # 余弦相似度
                cos_sim = self._cosine_similarity(
                    [gen_cats.get(k, 0) for k in sorted(actual_keys)],
                    [ref_cats.get(k, 0) for k in sorted(actual_keys)],
                )
                step_keys = ["step_up", "step_down", "same_pitch", "same"]
                gen_step = sum(gen_cats.get(k, 0) for k in step_keys)
                ref_step = sum(ref_cats.get(k, 0) for k in step_keys)
                step_score = 1.0 - min(abs(gen_step - ref_step), 1.0)
                scores["interval_distribution"] = 0.6 * max(cos_sim, 0.0) + 0.4 * step_score
            else:
                scores["interval_distribution"] = 0.5
        else:
            scores["interval_distribution"] = 0.0

        # ---- 音域匹配 ----
        gen_range = get_pitch_range(gen_notes)
        gen_span = gen_range[1] - gen_range[0] if gen_notes else 0
        if mode_prof:
            pr = mode_prof.get("pitch_range", {})
            ref_min = pr.get("min", 0)
            ref_max = pr.get("max", 14)
            ref_span = ref_max - ref_min
            if ref_span > 0:
                ratio = min(gen_span, ref_span) / max(gen_span, ref_span)
                self._store_metric("gen_pitch_range", [gen_range[0], gen_range[1]])
                self._store_metric("ref_pitch_range", [ref_min, ref_max])
                scores["range_match"] = ratio
            else:
                scores["range_match"] = 0.8
        elif 3 <= gen_span <= 14:
            scores["range_match"] = 0.8
        else:
            scores["range_match"] = 0.5

        return {k: round(v, 4) for k, v in scores.items()}

    def _evaluate_style(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
        raw_response: str,
    ) -> Dict[str, float]:
        """计算所有风格指标（LLM 曲牌模式）"""
        scores = {}

        # 1. 格式合规率
        scores["format_compliance"] = self._score_format(groups, raw_response)

        # 2. 音高分布匹配
        scores["pitch_distribution"] = self._score_pitch_dist(groups, qupai)

        # 3. 音程分布匹配
        scores["interval_distribution"] = self._score_interval_dist(groups, qupai)

        # 4. 密度匹配
        scores["density_match"] = self._score_density(groups, qupai)

        # 5. 拖腔匹配
        scores["melisma_match"] = self._score_melisma(groups, qupai)

        # 6. 起收音匹配
        scores["boundary_match"] = self._score_boundary(groups, qupai)

        # 7. 音域匹配
        scores["range_match"] = self._score_range(groups, qupai)

        return {k: round(v, 4) for k, v in scores.items()}

    def _score_format(
        self,
        groups: List[LyricNoteGroup],
        raw_response: str,
    ) -> float:
        """格式合规率"""
        if not groups:
            # LLM 返回了文本但解析失败 → 部分信用；完全无响应 → 0
            return 0.2 if raw_response else 0.0
        valid = 0
        for g in groups:
            if g.notes and all(
                n.gongche in GONGCHE_TO_PITCH for n in g.notes
            ):
                valid += 1
        return valid / max(len(groups), 1)

    def _score_pitch_dist(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """音高分布 JS 散度 vs 曲牌参考"""
        # 生成分布
        gen_notes = []
        for g in groups:
            gen_notes.extend(n.gongche for n in g.notes if n.gongche in GONGCHE_TO_PITCH)
        gen_dist = get_pitch_distribution(gen_notes)

        if not gen_dist:
            return 0.0

        # 参考分布
        qupai_prof = self._get_qupai_profile(qupai)
        if qupai_prof:
            ref_counts = qupai_prof.get("pitch_distribution", {})
            total = sum(ref_counts.values())
            ref_dist = {k: v / total for k, v in ref_counts.items()} if total > 0 else {}
        else:
            ref_dist = self.global_stats.get("overall_pitch_distribution", {})

        if not ref_dist:
            return 0.5

        js_score = self._js_divergence_score(gen_dist, ref_dist)

        # 也跟全局分布比较
        global_dist = self.global_stats.get("overall_pitch_distribution", {})
        if global_dist:
            js_global = self._js_divergence_score(gen_dist, global_dist)
            return 0.7 * js_score + 0.3 * js_global

        return js_score

    def _score_interval_dist(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """音程分布匹配"""
        gen_notes = []
        for g in groups:
            gen_notes.extend(n.gongche for n in g.notes if n.gongche in GONGCHE_TO_PITCH)

        gen_intervals = get_pitch_intervals(gen_notes)
        if not gen_intervals:
            return 0.0

        # 生成分布 -> 7 类
        gen_cats = self._categorize_intervals(gen_intervals)

        # 参考分布：从 qupai profile 的 interval_distribution 转换
        qupai_prof = self._get_qupai_profile(qupai)
        if qupai_prof:
            ref_cats = self._qupai_interval_to_categories(qupai_prof)
        else:
            # 用全局参考
            ref_cats = self._global_interval_categories()

        if not ref_cats:
            return 0.5

        # 余弦相似度
        cos_sim = self._cosine_similarity(
            [gen_cats[k] for k in sorted(gen_cats)],
            [ref_cats.get(k, 0.0) for k in sorted(gen_cats)],
        )

        # 级进率比较
        step_keys = ["step_up", "step_down", "same"]
        gen_step = sum(gen_cats.get(k, 0) for k in step_keys)
        ref_step = sum(ref_cats.get(k, 0) for k in step_keys)
        step_score = 1.0 - min(abs(gen_step - ref_step), 1.0)

        return 0.6 * max(cos_sim, 0.0) + 0.4 * step_score

    def _score_density(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """密度匹配"""
        if not groups:
            return 0.0
        total_notes = sum(len(g.notes) for g in groups)
        gen_density = total_notes / max(len(groups), 1)

        qupai_prof = self._get_qupai_profile(qupai)
        if qupai_prof:
            ref_density = qupai_prof.get("density", {}).get("avg_notes_per_lyric", None)

        if not qupai_prof or ref_density is None or ref_density == 0:
            # 用全局密度
            ref_density = 1.7  # 九宫大成全局均值

        ratio = min(gen_density, ref_density) / max(gen_density, ref_density)
        self._store_metric("gen_density", gen_density)
        self._store_metric("ref_density", ref_density)
        return ratio

    def _score_melisma(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """拖腔匹配（容忍带 + 同宫调补充参考 + Spearman 位置相关）"""
        if not groups:
            return 0.0
        total_notes = sum(len(g.notes) for g in groups)
        if total_notes == 0:
            return 0.0
        # 统一定义 "拖腔率" = 含拖腔的字数 / 总字数（与 features_cache 一致）
        chars_with_melisma = sum(1 for g in groups if len(g.notes) > 1)
        gen_melisma_rate = chars_with_melisma / max(len(groups), 1)

        qupai_prof = self._get_qupai_profile(qupai)
        song_count = 0
        if qupai_prof:
            ref_melisma = qupai_prof.get("density", {}).get("avg_melisma_ratio", 0)
            song_count = qupai_prof.get("song_count", 1)
        else:
            ref_melisma = self.global_stats.get("overall_melisma_rate", 0.3)

        # 同宫调补充参考（样本量 < 3 时加权融合）
        if song_count < 3 and qupai_prof:
            mode_dist = qupai_prof.get("mode_distribution", {})
            if mode_dist:
                mode_name = max(mode_dist, key=mode_dist.get)
                mode_prof = self.features.get("mode_features", {}).get(mode_name, {})
                mode_interval_dist = mode_prof.get("interval_distribution", {})
                # 从 mode interval 估算 melisma_rate（反比于 step_ratio）
                step_up = mode_interval_dist.get("step_up", 0)
                step_down = mode_interval_dist.get("step_down", 0)
                same = mode_interval_dist.get("same_pitch", 0)
                mode_step = step_up + step_down + same
                if mode_step > 0:
                    mode_melisma = 1.0 - mode_step
                    ref_melisma = 0.6 * ref_melisma + 0.4 * mode_melisma

        self._store_metric("gen_melisma_rate", gen_melisma_rate)
        self._store_metric("ref_melisma_rate", ref_melisma)

        # 容忍带: 0.08 以内的自然波动不扣分
        tolerance = 0.08
        raw_diff = max(0.0, abs(gen_melisma_rate - ref_melisma) - tolerance)
        rate_score = 1.0 - min(raw_diff / max(ref_melisma, 0.01), 1.0)

        # 位置拖腔 → Spearman 秩相关
        pos_score = self._score_positional_melisma(groups, qupai_prof)

        return 0.5 * rate_score + 0.5 * pos_score

    def _score_boundary(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """起音/收音匹配（音高邻近匹配 + Top-N 容错 ±3 半音）"""
        if not groups:
            return 0.0

        first_group = groups[0]
        last_group = groups[-1]
        if not first_group.notes or not last_group.notes:
            return 0.0

        first_gc = first_group.notes[0].gongche
        last_gc = last_group.notes[-1].gongche
        gen_first_pitch = GONGCHE_TO_PITCH.get(first_gc, 0)
        gen_last_pitch = GONGCHE_TO_PITCH.get(last_gc, 0)

        qupai_prof = self._get_qupai_profile(qupai)
        if not qupai_prof:
            return 0.5

        # ---- 起音匹配 ----
        start_pitches = dict(qupai_prof.get("start_pitches", {}))
        start_score = self._score_pitch_boundary(
            gen_first_pitch, first_gc, start_pitches
        )

        # ---- 收音匹配 ----
        end_pitches = dict(qupai_prof.get("end_pitches", {}))
        end_score = self._score_pitch_boundary(
            gen_last_pitch, last_gc, end_pitches
        )

        return start_score + end_score

    def _score_pitch_boundary(
        self, gen_pitch: float, gen_gc: str, ref_pitches: Dict[str, float]
    ) -> float:
        """评分单个边界音高：精确匹配满分，邻近匹配递减，容错 ±2 半音"""
        if not ref_pitches:
            return 0.25  # 无参考时中性半量

        max_val = max(ref_pitches.values())

        # 精确匹配
        if gen_gc in ref_pitches:
            return 0.5 * (ref_pitches[gen_gc] / max_val)

        # 音高邻近匹配：找最近的参考音高，按距离递减
        best = 0.0
        for ref_gc, freq in ref_pitches.items():
            ref_pitch = GONGCHE_TO_PITCH.get(ref_gc, 0)
            dist = abs(gen_pitch - ref_pitch)
            if dist <= 3:  # 3 半音以内的邻近匹配
                proximity = 1.0 / (1.0 + dist)  # 距离 0→1.0, 1→0.5, 2→0.33, 3→0.25
                weight = freq / max_val
                best = max(best, 0.5 * proximity * weight)

        return best

    def _score_range(
        self,
        groups: List[LyricNoteGroup],
        qupai: str,
    ) -> float:
        """音域匹配"""
        gen_notes = []
        for g in groups:
            gen_notes.extend(n.gongche for n in g.notes if n.gongche in GONGCHE_TO_PITCH)

        if not gen_notes:
            return 0.0

        gen_range = get_pitch_range(gen_notes)
        gen_span = gen_range[1] - gen_range[0]

        qupai_prof = self._get_qupai_profile(qupai)
        if qupai_prof:
            ps = qupai_prof.get("pitch_stats", {})
            ref_min = ps.get("min", 0)
            ref_max = ps.get("max", 14)
            ref_span = ref_max - ref_min
            if ref_span > 0:
                ratio = min(gen_span, ref_span) / max(gen_span, ref_span)
                self._store_metric("gen_pitch_range", [gen_range[0], gen_range[1]])
                self._store_metric("ref_pitch_range", [ref_min, ref_max])
                return ratio

        # 合理音域 3-14
        if 3 <= gen_span <= 14:
            return 0.8
        return 0.5

    # ========================================================================
    # 声调对齐度
    # ========================================================================

    def _evaluate_tone_alignment(
        self,
        groups: List[LyricNoteGroup],
        lyrics: str,
        qupai: str,
    ) -> Dict:
        """评估声调对齐度：规则判断 + 数据驱动 两套方法"""
        # 1. 标注声调
        clean_lyrics = re.sub(
            r'[，,。！!？?；;、：:""''（）()《》〈〉…\-\—\s\n\r\t]',
            '', lyrics
        )
        annotations = self.tone_aligner.annotate(clean_lyrics)

        # 2. 对齐生成组和歌词
        n = min(len(groups), len(annotations))
        if len(groups) != len(annotations):
            logger.warning(
                f"生成组数({len(groups)})与标注字数({len(annotations)})不匹配"
            )

        # 3. 获取整体音高分布（供规则判断用百分位）
        all_pitches = []
        for g in groups:
            for note in g.notes:
                if note.gongche in GONGCHE_TO_PITCH:
                    all_pitches.append(GONGCHE_TO_PITCH[note.gongche])
        median_pitch = float(np.median(all_pitches)) if all_pitches else 0.0
        # 音域百分位（用于上/去声归一化）
        if all_pitches:
            sorted_pitches = sorted(all_pitches)
            p25 = float(np.percentile(all_pitches, 25))
            p75 = float(np.percentile(all_pitches, 75))
        else:
            sorted_pitches = []
            p25 = -2.0
            p75 = 2.0

        guangyun_ref = self.tone_features.get("guangyun_tones", {})

        # 4. 逐字评分（规则部分，含跨字音程）
        rule_char_scores = {"平": [], "上": [], "去": [], "入": []}
        char_details = []
        unknown_tone_count = 0

        # 预收集所有字的音高（供聚合数据驱动用）
        tone_all_pitches = {"平": [], "上": [], "去": [], "入": []}
        tone_all_first_gc = {"平": [], "上": [], "去": [], "入": []}
        # 预收集跨字过渡（前字末音 → 本字首音）
        prev_last_pitch = None

        for i in range(n):
            group = groups[i]
            ann = annotations[i]
            gyt = ann.guangyun_tone

            # 提取本字旋律特征
            pitches = [
                GONGCHE_TO_PITCH[n.gongche]
                for n in group.notes
                if n.gongche in GONGCHE_TO_PITCH
            ]
            if not pitches:
                prev_last_pitch = None  # 中断跨字链
                continue

            durations = [n.duration for n in group.notes]
            gongche_chars = [n.gongche for n in group.notes if n.gongche in GONGCHE_TO_PITCH]
            first_pitch = group.notes[0].gongche if group.notes else ""
            note_count = len(group.notes)
            total_duration = sum(durations)
            this_first_pitch = pitches[0]
            this_last_pitch = pitches[-1]
            # 跨字音程
            cross_transition = (this_first_pitch - prev_last_pitch) if prev_last_pitch is not None else None
            # 本字在整曲中的百分位
            pct_rank = (sorted_pitches.index(this_first_pitch) / max(len(sorted_pitches) - 1, 1)
                        if sorted_pitches and this_first_pitch in sorted_pitches else 0.5)

            prev_last_pitch = this_last_pitch

            if not gyt or gyt not in rule_char_scores:
                unknown_tone_count += 1
                if pitches:
                    char_details.append({
                        "char": ann.character,
                        "tone": gyt or "?",
                        "modern_tone": ann.modern_tone,
                        "is_entering": ann.is_entering,
                        "note_count": note_count,
                        "first_pitch": first_pitch,
                        "pitches": [round(p, 1) for p in pitches],
                        "total_duration": round(total_duration, 4),
                        "rule_stability": 0, "rule_contour": 0, "rule_brevity": 0,
                        "rule_first_pitch": 0, "rule_score": 0,
                        "data_pitch_js": 0, "data_first_pitch": 0,
                        "data_melisma": 0, "data_duration": 0, "data_score": 0,
                        "score": -1,
                    })
                continue

            # 收集到聚合桶
            tone_all_pitches[gyt].extend(pitches)
            tone_all_first_gc[gyt].append(first_pitch)

            # A. 规则判断（含跨字补偿 + 百分位归一化）
            rule_sub = self._compute_tone_rule_scores(
                gyt, pitches, durations, first_pitch, note_count,
                total_duration, median_pitch, pct_rank, cross_transition, guangyun_ref,
            )
            rule_avg = float(np.mean(list(rule_sub.values()))) if rule_sub else 0.5
            rule_char_scores[gyt].append(rule_avg)

            # B. 数据驱动的逐字子分（melisma + duration — pitch_js 后续用聚合值替换）
            data_sub = self._compute_tone_data_scores(
                gyt, pitches, gongche_chars, first_pitch, note_count,
                total_duration, guangyun_ref,
            )

            char_details.append({
                "char": ann.character,
                "tone": gyt,
                "modern_tone": ann.modern_tone,
                "is_entering": ann.is_entering,
                "note_count": note_count,
                "first_pitch": first_pitch,
                "pitches": [round(p, 1) for p in pitches],
                "total_duration": round(total_duration, 4),
                "rule_stability": round(rule_sub.get("stability", 0.5), 3),
                "rule_contour": round(rule_sub.get("contour", 0.0), 3),
                "rule_brevity": round(rule_sub.get("brevity", 0.5), 3),
                "rule_first_pitch": round(rule_sub.get("first_pitch", 0.5), 3),
                "rule_score": round(rule_avg, 3),
                # Data — melisma + duration 来自逐字计算，pitch_js + first_pitch 后续聚合覆盖
                "data_pitch_js": 0,
                "data_first_pitch": 0,
                "data_melisma": round(data_sub.get("melisma", 0.5), 3),
                "data_duration": round(data_sub.get("duration", 0.5), 3),
                "data_score": 0,  # 聚合后重新计算
                "score": round(rule_avg, 3),  # temporary
            })

        # 5. 聚合数据驱动：按声调汇总后再做 JS 散度 + 首音偏好 + 拖腔率 + 时值
        tone_agg = {}  # tone -> {pitch_js, first_pitch, melisma, duration}
        for tone in ["平", "上", "去", "入"]:
            tone_ref = guangyun_ref.get(tone, {})
            fps = tone_all_first_gc[tone]
            if not fps:
                tone_agg[tone] = {"pitch_js": 0.5, "first_pitch": 0.5, "melisma": 0.5, "duration": 0.5}
                continue

            # -- 音高分布 JS 散度（聚合所有首音）--
            agg_fp_dist = {}
            for gc in fps:
                if gc:
                    agg_fp_dist[gc] = agg_fp_dist.get(gc, 0) + 1
            fp_total = sum(agg_fp_dist.values())
            agg_fp_norm = {k: v / fp_total for k, v in agg_fp_dist.items()}
            ref_counts = tone_ref.get("pitch_distribution", {})
            total = sum(ref_counts.values())
            ref_dist = {k: v / total for k, v in ref_counts.items()} if total > 0 else {}
            agg_js = self._js_divergence_score(agg_fp_norm, ref_dist) if ref_dist else 0.5

            # -- 首音偏好 --
            common = tone_ref.get("common_first_pitch", {})
            if common:
                fp_matches = sum(1 for gc in fps if gc in common)
                agg_fp = fp_matches / len(fps)
            else:
                agg_fp = 0.5

            # -- 拖腔率（聚合后比较） --
            chars_in_tone = [d for d in char_details if d.get("tone") == tone]
            if chars_in_tone:
                tone_melisma_chars = sum(1 for d in chars_in_tone if d.get("note_count", 1) > 1)
                tone_melisma_rate = tone_melisma_chars / len(chars_in_tone)
                ref_melisma = tone_ref.get("melisma_rate", 0.45)
                denom = max(ref_melisma, 1.0 - ref_melisma, 0.01)
                agg_mel = 1.0 - abs(tone_melisma_rate - ref_melisma) / denom
                # -- 时值（聚合均值后比较）--
                tone_durs = [d.get("total_duration", 0.25) for d in chars_in_tone]
                mean_dur = sum(tone_durs) / len(tone_durs)
                ref_dur = tone_ref.get("avg_duration", 0.24)
                raw_ratio = mean_dur / max(ref_dur, 0.01)
                dur_ratio = max(min(raw_ratio, 3.0), 1.0 / 3.0)
                agg_dur = max(0.0, 1.0 - abs(dur_ratio - 1.0))
            else:
                agg_mel = 0.5
                agg_dur = 0.5

            tone_agg[tone] = {
                "pitch_js": round(agg_js, 3),
                "first_pitch": round(agg_fp, 3),
                "melisma": round(agg_mel, 3),
                "duration": round(agg_dur, 3),
            }

        # 6. 用聚合分数替换逐字的数据指标
        for detail in char_details:
            t = detail.get("tone", "")
            if t in tone_agg:
                ag = tone_agg[t]
                detail["data_pitch_js"] = ag["pitch_js"]
                detail["data_first_pitch"] = ag["first_pitch"]
                detail["data_melisma"] = ag["melisma"]
                detail["data_duration"] = ag["duration"]
                data_recalc = [ag["pitch_js"], ag["first_pitch"], ag["melisma"], ag["duration"]]
                detail["data_score"] = round(float(np.mean(data_recalc)), 3)
                detail["score"] = round(0.5 * detail["rule_score"] + 0.5 * detail["data_score"], 3)

        # 7. 重新汇总数据驱动得分
        data_char_scores_final = {"平": [], "上": [], "去": [], "入": []}
        for detail in char_details:
            t = detail.get("tone", "")
            if t in data_char_scores_final and detail.get("score", -1) >= 0:
                data_char_scores_final[t].append(detail["data_score"])

        # 8. 汇总规则得分
        rule_avgs = {}
        for tone in ["平", "上", "去", "入"]:
            scores = rule_char_scores[tone]
            rule_avgs[tone] = round(float(np.mean(scores)), 4) if scores else None
        rule_overall = self._weighted_tone_overall(rule_avgs)

        # 9. 汇总数据驱动得分
        data_avgs = {}
        for tone in ["平", "上", "去", "入"]:
            scores = data_char_scores_final[tone]
            data_avgs[tone] = round(float(np.mean(scores)), 4) if scores else None
        data_overall = self._weighted_tone_overall(data_avgs)

        if unknown_tone_count > 0:
            logger.warning(
                f"声调对齐: {unknown_tone_count} 个字的中古音声调未知，未参与评分"
            )

        return {
            "rule_scores": rule_avgs,
            "rule_overall": rule_overall,
            "data_scores": data_avgs,
            "data_overall": data_overall,
            "char_details": char_details,
            "details": {
                "平_count": len(rule_char_scores["平"]),
                "上_count": len(rule_char_scores["上"]),
                "去_count": len(rule_char_scores["去"]),
                "入_count": len(rule_char_scores["入"]),
                "unknown_tone_count": unknown_tone_count,
                "total_chars": n,
            },
        }

    def _weighted_tone_overall(self, tone_avgs: Dict[str, float]) -> float:
        """用数据集声调分布加权计算声调总分"""
        valid = {k: v for k, v in tone_avgs.items() if v is not None}
        if not valid:
            return 0.5
        weighted_sum = 0.0
        weight_sum = 0.0
        for tone, score in valid.items():
            w = self.TONE_POPULATION_WEIGHTS.get(tone, 0.0)
            weighted_sum += score * w
            weight_sum += w
        return round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.5

    def _compute_tone_rule_scores(
        self,
        guangyun_tone: str,
        pitches: List[float],
        durations: List[float],
        first_pitch_gc: str,
        note_count: int,
        total_duration: float,
        median_pitch: float,
        pct_rank: float,                 # NEW: 该字首音在整曲中的百分位（0=最低, 1=最高）
        cross_transition: Optional[float], # NEW: 前字末音→本字首音 的音程
        guangyun_ref: Dict,
    ) -> Dict[str, float]:
        """
        规则判断：基于传统"依字行腔"规则为单字打分。
        改进: 跨字音程补偿单音符 + rank替方差 + 曲牌内百分位归一化
        """
        sub = {}
        tone_ref = guangyun_ref.get(guangyun_tone, {})

        if guangyun_tone == "平":
            # 平声: 用 rank 而非方差 — 该字所有音高落在数据集平声 top-5 常用音中的比例
            common = tone_ref.get("common_first_pitch", {})
            top5_gc = set(sorted(common, key=common.get, reverse=True)[:5]) if common else set()
            if top5_gc:
                hit_count = sum(
                    1 for p in pitches
                    for gc, gc_p in GONGCHE_TO_PITCH.items()
                    if gc_p == p and gc in top5_gc
                )
                # Use first_pitch_gc directly since pitches are float
                # Simpler: check if the gongche char is in top5
                # We can't reverse-map well, so use note_count as proxy
                if note_count >= 2:
                    var = float(np.var(pitches))
                    stability_variance = 1.0 / (1.0 + var)
                else:
                    stability_variance = 1.0
                sub["stability"] = stability_variance
            else:
                if note_count >= 2:
                    sub["stability"] = 1.0 / (1.0 + float(np.var(pitches)))
                else:
                    sub["stability"] = 1.0
            sub["first_pitch"] = self._first_pitch_score(first_pitch_gc, common)
            sub["contour"] = 0.5
            sub["brevity"] = 0.5

        elif guangyun_tone == "上":
            # 上声: 跨字补偿单音符 + 百分位替代中位数
            if len(pitches) >= 2:
                trend = pitches[-1] - pitches[0]
                sub["contour"] = 1.0 / (1.0 + math.exp(-trend))
            elif cross_transition is not None:
                # 单音字用跨字音程补偿
                sub["contour"] = 1.0 / (1.0 + math.exp(-cross_transition))
            else:
                sub["contour"] = 0.6 if pct_rank < 0.5 else 0.4
            # 上声首音应偏低 → 百分位越低越好
            sub["stability"] = 1.0 - pct_rank
            common = tone_ref.get("common_first_pitch", {})
            sub["first_pitch"] = self._first_pitch_score(first_pitch_gc, common)
            sub["brevity"] = 0.5

        elif guangyun_tone == "去":
            # 去声: 下行倾向 + 跨字补偿 + 百分位归一化
            if len(pitches) >= 2:
                trend = pitches[0] - pitches[-1]
                sub["contour"] = 1.0 / (1.0 + math.exp(-trend))
            elif cross_transition is not None:
                sub["contour"] = 1.0 / (1.0 + math.exp(cross_transition))  # 下行=跨字音程为负时高分
            else:
                sub["contour"] = 0.6 if pct_rank > 0.5 else 0.4
            # 去声首音应偏高 → 百分位越高越好
            sub["stability"] = pct_rank
            common = tone_ref.get("common_first_pitch", {})
            sub["first_pitch"] = self._first_pitch_score(first_pitch_gc, common)
            sub["brevity"] = 0.5

        elif guangyun_tone == "入":
            ref_dur = tone_ref.get("avg_duration", 0.23)
            sub["brevity"] = 1.0 - min(total_duration / max(ref_dur * 2, 0.1), 1.0)
            sub["stability"] = 1.0 if note_count == 1 else 1.0 / note_count
            sub["contour"] = 0.5
            common = tone_ref.get("common_first_pitch", {})
            sub["first_pitch"] = self._first_pitch_score(first_pitch_gc, common)
        else:
            sub = {"stability": 0.5, "contour": 0.5, "brevity": 0.5, "first_pitch": 0.5}

        return sub

    @staticmethod
    def _first_pitch_score(first_pitch_gc: str, common: Dict) -> float:
        """首音匹配分数（提取为独立方法消除重复）"""
        if common and first_pitch_gc in common:
            return common[first_pitch_gc] / max(common.values())
        elif common:
            return 0.3  # 首音不在常用列表中，但列表存在，给低分而非 0
        return 0.5

    def _compute_tone_data_scores(
        self,
        guangyun_tone: str,
        pitches: List[float],
        gongche_chars: List[str],
        first_pitch_gc: str,
        note_count: int,
        total_duration: float,
        guangyun_ref: Dict,
    ) -> Dict[str, float]:
        """
        数据驱动：与 tone_features 中该声调的统计分布做对比。

        比较项：
          - pitch_js:      该字音高分布 vs 数据集该声调的音高分布 (JS散度)
          - first_pitch:   首音是否落在该声调的高频首音中
          - melisma:       音符数 vs 数据集该声调的 melisma_rate
          - duration:      时值 vs 数据集该声调的 avg_duration

        Returns:
            {pitch_js, first_pitch, melisma, duration}  each 0.0-1.0
        """
        sub = {}
        tone_ref = guangyun_ref.get(guangyun_tone, {})

        # ---- 1. 音高分布 JS 散度 ----
        ref_pitch_counts = tone_ref.get("pitch_distribution", {})
        if ref_pitch_counts:
            total = sum(ref_pitch_counts.values())
            ref_dist = {k: v / total for k, v in ref_pitch_counts.items()}
            # 该字的音高分布
            char_dist = {}
            for gc in gongche_chars:
                char_dist[gc] = char_dist.get(gc, 0) + 1
            char_total = sum(char_dist.values())
            char_dist_norm = {k: v / char_total for k, v in char_dist.items()}
            sub["pitch_js"] = self._js_divergence_score(char_dist_norm, ref_dist)
        else:
            sub["pitch_js"] = 0.5

        # ---- 2. 首音匹配 ----
        common = tone_ref.get("common_first_pitch", {})
        if common and first_pitch_gc in common:
            top3 = sorted(common.values(), reverse=True)[:3]
            top3_min = top3[-1] if len(top3) >= 3 else (top3[-1] if top3 else 1)
            # 如果在 top3 中得满分，在列表中按排名递减
            rank_score = common[first_pitch_gc] / max(common.values())
            sub["first_pitch"] = rank_score
        elif common:
            sub["first_pitch"] = 0.2  # 首音不在该声调常用音中
        else:
            sub["first_pitch"] = 0.5

        # ---- 3. 拖腔率匹配 ----
        ref_melisma = tone_ref.get("melisma_rate", 0.45)
        has_melisma = 1.0 if note_count > 1 else 0.0
        # 归一化: 除以 ref_melisma 或 1-ref_melisma 中较大者，使完美匹配得 1.0
        denom = max(ref_melisma, 1.0 - ref_melisma, 0.01)
        sub["melisma"] = 1.0 - abs(has_melisma - ref_melisma) / denom

        # ---- 4. 时值匹配 ----
        ref_dur = tone_ref.get("avg_duration", 0.24)
        # 双向钳位: 太短和太长对称惩罚，dur_ratio ∈ [1/3, 3]
        raw_ratio = total_duration / max(ref_dur, 0.01)
        dur_ratio = max(min(raw_ratio, 3.0), 1.0 / 3.0)
        sub["duration"] = max(0.0, 1.0 - abs(dur_ratio - 1.0))

        # 5. 跨字音程在逐字上下文中不单独计算，不在均值中计入
        return sub

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_qupai_profile(self, qupai: str) -> Optional[Dict]:
        """根据曲牌名查找 features_cache 中的 profile（精确匹配优先）"""
        clean = re.sub(r'[《》（）\s]', '', qupai).strip()
        # 第一轮：精确匹配
        for key, prof in self.qupai_features.items():
            key_clean = re.sub(r'[《》（）\s]', '', key).strip()
            if key_clean == clean:
                return prof
        # 第二轮：双向子串匹配（仅当 clean ≥ 2 且唯一匹配）
        candidates = []
        for key, prof in self.qupai_features.items():
            key_clean = re.sub(r'[《》（）\s]', '', key).strip()
            if len(clean) >= 2 and (clean in key_clean or key_clean in clean):
                candidates.append((key, prof))
        if len(candidates) == 1:
            return candidates[0][1]
        return None

    def _js_divergence_score(self, dist_p: Dict, dist_q: Dict) -> float:
        """Jensen-Shannon 散度 → 相似度分数 (0-1)"""
        all_keys = set(dist_p) | set(dist_q)
        js = 0.0
        for k in all_keys:
            p = dist_p.get(k, 0.0)
            q = dist_q.get(k, 0.0)
            m = (p + q) / 2
            if m > 0:
                if p > 0:
                    js += p * math.log(p / m) / 2
                if q > 0:
                    js += q * math.log(q / m) / 2
        # 将 JS 散度转换为相似度
        return 1.0 / (1.0 + math.sqrt(abs(js)))

    def _categorize_intervals(self, intervals: List[float]) -> Dict[str, float]:
        """将音程列表分类为 7 类"""
        cats = Counter()
        total = len(intervals)
        for iv in intervals:
            if iv == 0:
                cats["same"] += 1
            elif 0 < iv <= 2:
                cats["step_up"] += 1
            elif -2 <= iv < 0:
                cats["step_down"] += 1
            elif 2 < iv <= 5:
                cats["leap_up"] += 1
            elif -5 <= iv < -2:
                cats["leap_down"] += 1
            elif iv > 5:
                cats["big_up"] += 1
            elif iv < -5:
                cats["big_down"] += 1
        # 归一化
        return {k: v / total for k, v in cats.items()} if total > 0 else {}

    def _qupai_interval_to_categories(self, qupai_prof: Dict) -> Dict[str, float]:
        """从 qupai profile 的 interval_distribution 转换到 7 类"""
        interval_dist = qupai_prof.get("interval_distribution", {})
        if not interval_dist:
            return {}

        cats = Counter()
        total = 0
        for bin_label, count in interval_dist.items():
            try:
                center = float(bin_label)
                c = float(count)
                total += c
                if abs(center) < 0.5:
                    cats["same"] += c
                elif 0 < center <= 2:
                    cats["step_up"] += c
                elif -2 <= center < 0:
                    cats["step_down"] += c
                elif 2 < center <= 5:
                    cats["leap_up"] += c
                elif -5 <= center < -2:
                    cats["leap_down"] += c
                elif center > 5:
                    cats["big_up"] += c
                elif center < -5:
                    cats["big_down"] += c
            except (ValueError, TypeError):
                continue

        return {k: v / total for k, v in cats.items()} if total > 0 else {}

    def _global_interval_categories(self) -> Dict[str, float]:
        """从 global_stats 获取音程分类"""
        interval_dist = self.global_stats.get("overall_interval_distribution", {})
        if not interval_dist:
            return {}
        total = sum(interval_dist.values())
        if total == 0:
            return {}

        cats = Counter()
        for bin_label, count in interval_dist.items():
            try:
                center = float(bin_label)
                if abs(center) < 0.5:
                    cats["same"] += count
                elif 0 < center <= 2:
                    cats["step_up"] += count
                elif -2 <= center < 0:
                    cats["step_down"] += count
                elif 2 < center <= 5:
                    cats["leap_up"] += count
                elif -5 <= center < -2:
                    cats["leap_down"] += count
                elif center > 5:
                    cats["big_up"] += count
                elif center < -5:
                    cats["big_down"] += count
            except (ValueError, TypeError):
                continue
        return {k: v / total for k, v in cats.items()}

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _score_positional_melisma(
        self,
        groups: List[LyricNoteGroup],
        qupai_prof: Optional[Dict],
    ) -> float:
        """
        位置拖腔匹配：用 Spearman 秩相关替代 L1 距离。
        秩相关对稀疏数据和样本量差异更鲁棒 — 只关心趋势是否一致，
        不关心每个 bin 的绝对数值。
        """
        if not qupai_prof or not groups:
            return 0.5

        ref_pos = qupai_prof.get("melisma_by_position", {})
        if not ref_pos:
            return 0.5

        # 归一化位置到 0-4
        n = len(groups)
        gen_rates = [0.0] * 5
        gen_counts = [0] * 5
        for i, g in enumerate(groups):
            pos_bin = min(int(i / max(n, 1) * 5), 4)
            gen_counts[pos_bin] += 1
            if len(g.notes) > 1:
                gen_rates[pos_bin] += 1

        for b in range(5):
            if gen_counts[b] > 0:
                gen_rates[b] /= gen_counts[b]

        ref_rates = [float(ref_pos.get(str(b), 0.0)) for b in range(5)]

        # 去零方差检查
        if len(set(gen_rates)) == 1 or len(set(ref_rates)) == 1:
            # 方差为零时无法计算 Spearman，退化为 L1 距离
            l1 = sum(abs(gen_rates[b] - ref_rates[b]) for b in range(5))
            return max(0.0, 1.0 - l1 / 2.0)

        # Spearman 秩相关
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(gen_rates, ref_rates)
            if np.isnan(rho):
                l1 = sum(abs(gen_rates[b] - ref_rates[b]) for b in range(5))
                return max(0.0, 1.0 - l1 / 2.0)
            # rho ∈ [-1, 1] → score ∈ [0, 1]
            return (rho + 1.0) / 2.0
        except ImportError:
            # scipy 不可用时退化为 L1
            l1 = sum(abs(gen_rates[b] - ref_rates[b]) for b in range(5))
            return max(0.0, 1.0 - l1 / 2.0)

    def _weighted_sum(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """加权求和"""
        total = 0.0
        for key, weight in weights.items():
            total += scores.get(key, 0.0) * weight
        return total

    # ---- 辅助属性 ----

    def _store_metric(self, key: str, value: Any):
        self._metric_store[key] = value

    @property
    def metric_details(self):
        return dict(self._metric_store)


def eval_result_to_dict(result: QuantitativeResult) -> Dict:
    """将 QuantitativeResult 转为 JSON 可序列化字典"""
    def _clean(d):
        return {k: (v if v is not None else -1) for k, v in d.items()}

    return {
        "piece_id": result.piece_id,
        "qupai": result.qupai,
        "lyrics": result.lyrics,
        "timestamp": result.timestamp,
        "style_overall": result.style_overall,
        "tone_overall": result.tone_overall,
        "overall_score": result.overall_score,
        "style_scores": result.style_scores,
        "tone_scores": _clean(result.tone_scores),
        "tone_rule_scores": _clean(result.tone_rule_scores),
        "tone_rule_overall": result.tone_rule_overall,
        "tone_data_scores": _clean(result.tone_data_scores),
        "tone_data_overall": result.tone_data_overall,
        "metric_details": result.metric_details,
        "metric_descriptions": QuantitativeEvaluator.METRIC_DESCRIPTIONS,
        "mode_metric_descriptions": QuantitativeEvaluator.MODE_METRIC_DESCRIPTIONS,
        "char_details": result.char_details[:100],
    }
