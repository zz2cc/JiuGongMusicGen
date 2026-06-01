"""
评估系统模块 — 对生成的工尺谱旋律进行客观指标评估。

评估维度：
1. 音高覆盖率 — 是否使用了该曲牌的典型音高
2. 音域匹配 — 生成旋律的音域是否合理
3. 板拍密度 — 每字对应音符数是否合理
4. 格式合规率 — 输出是否可解析
5. 起音/收音 — 首音与末音是否符合曲牌惯例
6. 音程分布 — 音程跳跃模式是否匹配
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np

from .data_loader import JiuGongDataset, Song, NoteEvent, LyricNoteGroup
from .gongche_vocab import (
    GONGCHE_TO_PITCH, get_pitch_distribution,
    get_interval_distribution, melody_similarity,
    get_pitch_range, get_pitch_intervals,
)
from .generator import GeneratedPiece

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """单次评估结果"""
    piece_id: str
    scores: Dict[str, float]
    details: Dict[str, any]
    overall_score: float


class QupaiEvaluator:
    """
    曲牌音乐生成质量评估器。

    Usage:
        evaluator = QupaiEvaluator(dataset, index)
        result = evaluator.evaluate(generated_piece, reference_qupai="奉時春")
        print(result.overall_score)
    """

    def __init__(self, dataset: JiuGongDataset, index=None):
        self.dataset = dataset
        self.index = index

    def evaluate(
        self,
        piece: GeneratedPiece,
        reference_qupai: str = None,
        reference_song_ids: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        评估生成的旋律。

        Args:
            piece: 生成的结果
            reference_qupai: 参考曲牌（用于对比统计）
            reference_song_ids: 参考歌曲ID列表

        Returns:
            EvaluationResult
        """
        if reference_qupai is None:
            reference_qupai = piece.qupai

        scores = {}
        details = {}

        # ---- 1. 格式合规率 ----
        scores["format_compliance"] = self._score_format_compliance(piece)

        # ---- 2. 音高覆盖率 ----
        scores["pitch_coverage"] = self._score_pitch_coverage(
            piece, reference_qupai
        )

        # ---- 3. 音域匹配 ----
        scores["pitch_range_match"] = self._score_range_match(
            piece, reference_qupai
        )

        # ---- 4. 板拍密度 ----
        scores["density_match"] = self._score_density_match(
            piece, reference_qupai
        )

        # ---- 5. 起音/收音 ----
        scores["boundary_match"] = self._score_boundary_match(
            piece, reference_qupai
        )

        # ---- 6. 音程分布 ----
        scores["interval_distribution"] = self._score_interval_match(
            piece, reference_qupai
        )

        # ---- 7. 与参考样例的相似度 ----
        if reference_song_ids:
            scores["reference_similarity"] = self._score_reference_similarity(
                piece, reference_song_ids
            )
        else:
            scores["reference_similarity"] = 0.5  # 无参考时中性分

        # ---- 综合评分 ----
        weights = {
            "format_compliance": 0.20,
            "pitch_coverage": 0.25,
            "pitch_range_match": 0.10,
            "density_match": 0.15,
            "boundary_match": 0.10,
            "interval_distribution": 0.10,
            "reference_similarity": 0.10,
        }
        overall = sum(scores.get(k, 0) * w for k, w in weights.items())

        return EvaluationResult(
            piece_id=f"{piece.qupai}_{piece.lyrics[:6]}",
            scores=scores,
            details=details,
            overall_score=round(overall, 4),
        )

    # ========================================================================
    # 评分方法
    # ========================================================================

    def _score_format_compliance(self, piece: GeneratedPiece) -> float:
        """格式合规率: 输出是否有有效的工尺谱旋律"""
        if not piece.success:
            return 0.0
        if not piece.melody:
            return 0.0
        # 每个分组至少有一个音符（含有效的工尺字符）
        valid_groups = 0
        for g in piece.melody:
            if g.notes and all(
                n.gongche in GONGCHE_TO_PITCH for n in g.notes
            ):
                valid_groups += 1
        return valid_groups / max(len(piece.melody), 1)

    def _score_pitch_coverage(
        self, piece: GeneratedPiece, qupai: str
    ) -> float:
        """
        音高覆盖率: 生成旋律的工尺分布与目标曲牌参考分布之间的相似度。

        使用 1 - JS 散度 作为相似度。
        """
        # 生成旋律的音高分布
        gen_notes = []
        for g in piece.melody:
            gen_notes.extend(n.gongche for n in g.notes)
        gen_dist = get_pitch_distribution(gen_notes)

        # 参考分布（从索引中获取）
        if self.index:
            profile = self.index.get_profile(qupai)
            if profile and profile.pitch_distribution:
                ref_dist = profile.pitch_distribution

                # JS 散度
                all_gc = set(gen_dist) | set(ref_dist)
                js = 0.0
                for gc in all_gc:
                    p = gen_dist.get(gc, 0.0)
                    q = ref_dist.get(gc, 0.0)
                    m = (p + q) / 2
                    if m > 0:
                        if p > 0:
                            js += p * np.log(p / m) / 2
                        if q > 0:
                            js += q * np.log(q / m) / 2

                return float(1.0 / (1.0 + np.sqrt(abs(js))))

        return 0.5  # 无参考时中性分

    def _score_range_match(
        self, piece: GeneratedPiece, qupai: str
    ) -> float:
        """
        音域匹配: 生成旋律的音域是否在合理范围内。
        """
        gen_notes = []
        for g in piece.melody:
            gen_notes.extend(n.gongche for n in g.notes)

        gen_range = get_pitch_range(gen_notes)
        gen_span = gen_range[1] - gen_range[0]

        if self.index:
            profile = self.index.get_profile(qupai)
            if profile:
                ref_lo, ref_hi = profile.avg_pitch_range
                ref_span = ref_hi - ref_lo
                if ref_span > 0:
                    # 音域跨度比
                    ratio = min(gen_span, ref_span) / max(gen_span, ref_span)
                    return ratio

        # 合理音域: 3 ~ 14 (3个八度以内)
        if 3 <= gen_span <= 14:
            return 0.8
        return 0.5

    def _score_density_match(
        self, piece: GeneratedPiece, qupai: str
    ) -> float:
        """
        板拍密度: 每字音符数与曲牌平均值比较。
        """
        if not piece.melody or not piece.lyrics:
            return 0.5

        total_notes = sum(len(g.notes) for g in piece.melody)
        total_lyrics = len(piece.melody)
        gen_density = total_notes / max(total_lyrics, 1)

        if self.index:
            profile = self.index.get_profile(qupai)
            if profile and profile.avg_notes_per_lyric > 0:
                ref_density = profile.avg_notes_per_lyric
                # 密度比
                ratio = min(gen_density, ref_density) / max(gen_density, ref_density)
                return ratio

        # 一般合理范围: 1.0 ~ 3.0
        if 1.0 <= gen_density <= 3.0:
            return 0.8
        elif 0.5 <= gen_density <= 5.0:
            return 0.5
        return 0.2

    def _score_boundary_match(
        self, piece: GeneratedPiece, qupai: str
    ) -> float:
        """
        起音/收音匹配: 首音和末音是否使用了该曲牌的常用起收音。
        """
        if not piece.melody:
            return 0.0

        # 首字首音
        first_group = piece.melody[0]
        if not first_group.notes:
            return 0.0
        first_gongche = first_group.notes[0].gongche

        # 末字末音
        last_group = piece.melody[-1]
        if not last_group.notes:
            return 0.0
        last_gongche = last_group.notes[-1].gongche

        score = 0.0

        if self.index:
            profile = self.index.get_profile(qupai)
            if profile:
                # 起音匹配
                if profile.common_start_pitches:
                    if first_gongche in profile.common_start_pitches:
                        # 加权: 最常见起音得高分
                        score += 0.5 * profile.common_start_pitches.get(
                            first_gongche, 0.0
                        ) / max(profile.common_start_pitches.values())

                # 收音匹配
                if profile.common_end_pitches:
                    if last_gongche in profile.common_end_pitches:
                        score += 0.5 * profile.common_end_pitches.get(
                            last_gongche, 0.0
                        ) / max(profile.common_end_pitches.values())

                return score

        return 0.5  # 无参考时中性分

    def _score_interval_match(
        self, piece: GeneratedPiece, qupai: str
    ) -> float:
        """
        音程分布匹配: 比较生成旋律的音程分布与参考。
        """
        gen_notes = []
        for g in piece.melody:
            gen_notes.extend(n.gongche for n in g.notes)

        gen_intervals = get_pitch_intervals(gen_notes)
        gen_interval_dist = {}
        if gen_intervals:
            counter = Counter(gen_intervals)
            total = len(gen_intervals)
            gen_interval_dist = {k: v / total for k, v in counter.items()}

        # 与数据集整体音程分布比较（简化：评分合理性与曲牌无关）
        # 九宫大成的典型特征是五声音阶级进为主（±2），跳进为辅
        if not gen_interval_dist:
            return 0.0

        # 检查是否以级进为主（级进比例应 > 50%）
        step_intervals = sum(
            v for k, v in gen_interval_dist.items() if abs(k) <= 2.0
        )
        leap_intervals = sum(
            v for k, v in gen_interval_dist.items() if 2.0 < abs(k) <= 5.0
        )

        step_score = min(step_intervals / 0.6, 1.0)  # 60%级进=满分
        leap_score = 1.0 - abs(leap_intervals - 0.3)  # ~30%跳进正常

        return (step_score * 0.6 + leap_score * 0.4)

    def _score_reference_similarity(
        self, piece: GeneratedPiece, reference_ids: List[str]
    ) -> float:
        """
        与参考样例的旋律相似度。
        """
        gen_seq = []
        for g in piece.melody:
            gen_seq.extend(n.gongche for n in g.notes if n.gongche)

        similarities = []
        for pid in reference_ids[:3]:  # 最多对比3首
            try:
                ref_song = self.dataset.get_song(pid)
                ref_seq = [n.gongche for n in ref_song.notes if n.gongche]
                sim = melody_similarity(gen_seq, ref_seq)
                similarities.append(sim)
            except (KeyError, Exception):
                continue

        if similarities:
            return np.mean(similarities)
        return 0.5

    # ========================================================================
    # 批量评估
    # ========================================================================

    def evaluate_batch(
        self,
        pieces: List[GeneratedPiece],
        reference_qupai: str = None,
    ) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for piece in pieces:
            result = self.evaluate(piece, reference_qupai=reference_qupai)
            results.append(result)
        return results

    def compare_variants(
        self,
        variant_results: Dict[str, List[GeneratedPiece]],
    ) -> Dict:
        """
        比较不同实验变体的评估结果。

        Args:
            variant_results: {variant_name: [GeneratedPiece, ...], ...}

        Returns:
            {
                "summary": {variant_name: {avg_score, avg_*}, ...},
                "per_piece": [...],
                "best_variant": str,
            }
        """
        summary = {}
        per_piece = []

        for variant, pieces in variant_results.items():
            evals = self.evaluate_batch(pieces)
            scores_list = [e.overall_score for e in evals]
            metric_avgs = {}
            if evals:
                for key in evals[0].scores:
                    metric_avgs[f"avg_{key}"] = float(
                        np.mean([e.scores[key] for e in evals])
                    )
                per_piece.extend(evals)

            summary[variant] = {
                "avg_overall_score": float(np.mean(scores_list)) if scores_list else 0,
                "success_rate": sum(1 for p in pieces if p.success) / max(len(pieces), 1),
                **metric_avgs,
            }

        best_variant = max(summary, key=lambda v: summary[v]["avg_overall_score"])

        return {
            "summary": summary,
            "per_piece": per_piece,
            "best_variant": best_variant,
        }

    def save_report(
        self,
        results: Dict,
        output_path: str,
    ):
        """保存评估报告"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 将不可序列化的对象转为基本类型
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"评估报告已保存: {output_path}")
