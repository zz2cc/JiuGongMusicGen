"""
数据集特征提取器 — 一次性从 CSV 计算所有统计特征，缓存为 JSON。

提取的特征：
1. 曲牌画像 (per-qupai): 起音收音、音高分布、拖腔密度、音程模式
2. 声调-旋律对齐 (tone-melody): 平上去入→典型工尺和音高走向
3. 宫调特征 (per-mode): 音域、骨架音、节奏模式
4. 南词vs北词风格对比
"""

import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from .gongche_vocab import GONGCHE_TO_PITCH, GONGCHE_CHARS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """一次性特征提取器，结果可缓存"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.songs_csv = self.data_dir / "FINAL_SONGS.csv"
        self.notes_csv = self.data_dir / "FINAL_NOTES.csv"
        self.beats_csv = self.data_dir / "FINAL_BEATS.csv"

        # 缓存
        self.qupai_features: Dict = {}      # 曲牌→特征
        self.tone_features: Dict = {}        # 声调→旋律特征
        self.mode_features: Dict = {}        # 宫调→特征
        self.region_features: Dict = {}      # 南词/北词→特征
        self.global_stats: Dict = {}         # 全局统计

    def extract_all(self, cache_path: Optional[str] = None) -> Dict:
        """运行所有特征提取，返回完整特征字典"""
        t0 = time.time()
        logger.info("开始特征提取...")

        # 加载 CSV
        songs_df = pd.read_csv(self.songs_csv, encoding="utf-8")
        notes_df = pd.read_csv(self.notes_csv, encoding="utf-8", low_memory=False)
        beats_df = pd.read_csv(self.beats_csv, encoding="utf-8", low_memory=False)

        # 类型转换
        notes_df["polyu_id"] = notes_df["polyu_id"].astype(str)
        notes_df["gongche_pitch"] = pd.to_numeric(notes_df["gongche_pitch"], errors="coerce")
        notes_df["duration"] = pd.to_numeric(notes_df["duration"], errors="coerce")
        notes_df["multi_rhythm_num"] = pd.to_numeric(notes_df["multi_rhythm_num"], errors="coerce").fillna(0).astype(int)
        songs_df["polyu_id"] = songs_df["polyu_id"].astype(str)

        logger.info(f"  歌曲: {len(songs_df)}, 音符: {len(notes_df)}, 板拍: {len(beats_df)}")

        # --- 合并数据 ---
        merged = notes_df.merge(
            songs_df[["polyu_id", "qupai_brackets_removed", "volume_region",
                       "volume_mode", "note_count"]],
            on="polyu_id", how="left"
        )

        # --- 1. 曲牌画像 ---
        logger.info("  1/4 提取曲牌画像...")
        self.qupai_features = self._extract_qupai_features(merged)

        # --- 2. 声调-旋律对齐 ---
        logger.info("  2/4 提取声调-旋律对齐...")
        self.tone_features = self._extract_tone_features(merged)

        # --- 3. 宫调特征 ---
        logger.info("  3/4 提取宫调特征...")
        self.mode_features = self._extract_mode_features(merged, beats_df)

        # --- 4. 南词vs北词 ---
        logger.info("  4/4 提取南词北词风格...")
        self.region_features = self._extract_region_features(merged)

        # --- 全局统计 ---
        self.global_stats = self._extract_global_stats(merged)

        result = {
            "qupai_features": self.qupai_features,
            "tone_features": self.tone_features,
            "mode_features": self.mode_features,
            "region_features": self.region_features,
            "global_stats": self.global_stats,
        }

        if cache_path:
            self._save_cache(result, cache_path)

        elapsed = time.time() - t0
        logger.info(f"特征提取完成！耗时 {elapsed:.0f}s, "
                     f"曲牌: {len(self.qupai_features)}, "
                     f"宫调: {len(self.mode_features)}")
        return result

    # ==================================================================
    # 1. 曲牌画像
    # ==================================================================

    def _extract_qupai_features(self, merged: pd.DataFrame) -> Dict[str, dict]:
        """按曲牌聚合，提取旋律结构特征"""
        features = {}

        for qupai, group in merged.groupby("qupai_brackets_removed"):
            if pd.isna(qupai) or str(qupai) == "nan" or len(group) < 1:
                continue

            qupai = str(qupai)
            songs_in_qupai = group["polyu_id"].nunique()

            # (a) 起音分布 — 每首歌第一个音符的工尺
            first_notes = group.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"]) \
                               .groupby("polyu_id").first()
            start_pitches = first_notes["gongche"].value_counts().to_dict()
            start_pitch_vals = first_notes["gongche_pitch"].describe().to_dict()

            # (b) 收音分布 — 每首歌最后一个音符的工尺
            last_notes = group.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"]) \
                              .groupby("polyu_id").last()
            end_pitches = last_notes["gongche"].value_counts().to_dict()
            end_pitch_vals = last_notes["gongche_pitch"].describe().to_dict()

            # (c) 整体音高分布
            pitch_dist = group["gongche"].value_counts().to_dict()
            pitch_stats = {
                "mean": float(group["gongche_pitch"].mean()),
                "std": float(group["gongche_pitch"].std()),
                "min": float(group["gongche_pitch"].min()),
                "max": float(group["gongche_pitch"].max()),
                "median": float(group["gongche_pitch"].median()),
            }

            # (d) 拖腔密度 — 每字音符数
            # lyric_id=0 聚合 → 每首歌的 (歌词字数, 总音符数)
            per_song = group.groupby("polyu_id").agg(
                total_notes=("gongche", "count"),
                unique_lyrics=("lyric_id", "nunique"),  # 近似
                melisma_notes=("multi_rhythm_num", lambda x: (x > 0).sum()),
            )
            per_song["density"] = per_song["total_notes"] / per_song["unique_lyrics"].clip(lower=1)
            per_song["melisma_ratio"] = per_song["melisma_notes"] / per_song["total_notes"].clip(lower=1)

            density_stats = {
                "avg_notes_per_lyric": float(per_song["density"].mean()),
                "std_notes_per_lyric": float(per_song["density"].std()),
                "avg_melisma_ratio": float(per_song["melisma_ratio"].mean()),
                "max_notes_per_lyric": float(per_song["density"].max()),
            }

            # (e) 音程模式 — 相邻音符间 Pitch 差值分布
            sorted_group = group.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"])
            pitches = sorted_group["gongche_pitch"].values
            pid_values = sorted_group["polyu_id"].values
            intervals = np.diff(pitches)
            # 去掉跨歌曲的间隔（polyu_id 变化点的 interval 是假的）
            cross_song_mask = (pid_values[:-1] != pid_values[1:])
            valid_intervals = intervals[~cross_song_mask]
            interval_dist = self._safe_interval_distribution(valid_intervals)

            # (f) 节奏特征
            dur_dist = group["duration"].value_counts().to_dict()
            dur_dist = {str(k): int(v) for k, v in dur_dist.items()}

            # (g) 南词/北词 占比
            region_dist = group["volume_region"].value_counts().to_dict()
            region_dist = {str(k): int(v) for k, v in region_dist.items() if pd.notna(k)}

            # (h) 宫调分布
            mode_dist = group["volume_mode"].value_counts().to_dict()
            mode_dist = {str(k): int(v) for k, v in mode_dist.items() if pd.notna(k)}

            # (i) 拖腔位置分布 — 按句中归一化位置统计拖腔概率
            sorted_q = group.sort_values(["polyu_id", "line_id", "lyric_id", "gongche_id"])
            # Count notes per (polyu_id, line_id, lyric_id) — all at once
            note_counts = sorted_q.groupby(["polyu_id", "line_id", "lyric_id"]).size()
            # Only first notes (multi_rhythm_num == 0) matter for position
            first_notes = sorted_q[sorted_q["multi_rhythm_num"] == 0].copy()
            first_notes["_has_melisma"] = False
            # Iterate rows to mark melisma (one pass through first_notes)
            for idx, row in first_notes.iterrows():
                key = (row["polyu_id"], row["line_id"], row["lyric_id"])
                first_notes.at[idx, "_has_melisma"] = note_counts.get(key, 1) > 1

            pos_bins = {0: [], 1: [], 2: [], 3: [], 4: []}
            for (_pid, _lid), line_grp in first_notes.groupby(["polyu_id", "line_id"]):
                n = len(line_grp)
                if n < 2: continue
                row_indices = list(line_grp.index)
                for i, idx in enumerate(row_indices):
                    # Always map last position to bin 4 (句尾), others to 0-3 proportionally
                    if i == n - 1:
                        bin_idx = 4
                    else:
                        bin_idx = min(int(i * 4 / (n - 1)), 3)
                    has_mel = first_notes.at[idx, "_has_melisma"]
                    pos_bins[bin_idx].append(has_mel)
            melisma_by_position = {}
            for bi in range(5):
                if pos_bins[bi]:
                    melisma_by_position[str(bi)] = round(
                        sum(pos_bins[bi]) / len(pos_bins[bi]), 3
                    )

            features[qupai] = {
                "song_count": int(songs_in_qupai),
                "total_notes": int(len(group)),
                "start_pitches": {str(k): int(v) for k, v in start_pitches.items()},
                "end_pitches": {str(k): int(v) for k, v in end_pitches.items()},
                "start_pitch_mean": float(start_pitch_vals.get("mean", 0)),
                "end_pitch_mean": float(end_pitch_vals.get("mean", 0)),
                "pitch_distribution": {str(k): int(v) for k, v in pitch_dist.items()},
                "pitch_stats": pitch_stats,
                "density": density_stats,
                "interval_distribution": interval_dist,
                "duration_distribution": dur_dist,
                "melisma_by_position": melisma_by_position,
                "region_distribution": region_dist,
                "mode_distribution": mode_dist,
            }

        return features

    # ==================================================================
    # 2. 声调-旋律对齐
    # ==================================================================

    def _extract_tone_features(self, merged: pd.DataFrame) -> Dict:
        """分析不同声调对应的旋律特征"""
        features = {}

        # --- 现代声调 (1=阴平 2=阳平 3=上声 4=去声) ---
        modern_tone_features = {}
        for tone_val, tone_name in [(1, "阴平"), (2, "阳平"), (3, "上声"), (4, "去声"), (0, "轻声")]:
            subset = merged[merged["lyric_modern_tone"] == tone_val]
            if len(subset) == 0:
                continue

            # 音高分布
            pitch_dist = subset["gongche"].value_counts().head(10).to_dict()

            # 作为歌词首音时的工尺（即 multi_rhythm_num==0 的）
            first_notes = subset[subset["multi_rhythm_num"] == 0]
            first_pitch = first_notes["gongche"].value_counts().head(5).to_dict()

            # 音高走向：看从 multi_rhythm_num==0 到下一个音符的差值
            tone_transitions = []
            for pid in subset["polyu_id"].unique()[:500]:  # 采样防溢出
                pid_data = subset[subset["polyu_id"] == pid].sort_values(
                    ["beat_id", "lyric_id", "gongche_id"]
                )
                pitches = pid_data["gongche_pitch"].values
                if len(pitches) >= 2:
                    # 找 lyric_id 变化点（新字开始）
                    lyrics = pid_data["lyric_id"].values
                    for i in range(len(lyrics) - 1):
                        if lyrics[i] != lyrics[i+1]:
                            tone_transitions.append(float(pitches[i+1] - pitches[i]))

            avg_transition = float(np.mean(tone_transitions)) if tone_transitions else 0.0

            modern_tone_features[tone_name] = {
                "count": int(len(subset)),
                "pitch_distribution": {str(k): int(v) for k, v in pitch_dist.items()},
                "common_first_pitch": {str(k): int(v) for k, v in first_pitch.items()},
                "avg_transition_to_next": round(avg_transition, 2),
                "transition_samples": len(tone_transitions),
            }

        features["modern_tones"] = modern_tone_features

        # --- 广韵声调 (平/上/去/入) ---
        guangyun_features = {}
        for tone_name in ["平", "上", "去", "入"]:
            subset = merged[merged["lyric_guangyun_tone"] == tone_name]
            if len(subset) == 0:
                continue

            pitch_dist = subset["gongche"].value_counts().head(10).to_dict()
            first_notes = subset[subset["multi_rhythm_num"] == 0]
            first_pitch = first_notes["gongche"].value_counts().head(5).to_dict()

            # 拖腔统计
            melisma_rate = (subset["multi_rhythm_num"] > 0).mean()

            # 入声字是否更短
            avg_duration = float(subset["duration"].mean())

            guangyun_features[tone_name] = {
                "count": int(len(subset)),
                "pitch_distribution": {str(k): int(v) for k, v in pitch_dist.items()},
                "common_first_pitch": {str(k): int(v) for k, v in first_pitch.items()},
                "melisma_rate": round(float(melisma_rate), 3),
                "avg_duration": round(avg_duration, 4),
            }

        features["guangyun_tones"] = guangyun_features

        return features

    # ==================================================================
    # 3. 宫调特征
    # ==================================================================

    def _extract_mode_features(self, merged: pd.DataFrame,
                                beats_df: pd.DataFrame) -> Dict[str, dict]:
        """提取73种宫调的音高和节奏特征"""
        features = {}

        for mode, group in merged.groupby("volume_mode"):
            if pd.isna(mode) or str(mode) == "nan" or len(group) < 1:
                continue
            mode = str(mode)

            # --- 音高特征 ---
            pitch_vals = group["gongche_pitch"].dropna()
            pitch_range = {
                "min": float(pitch_vals.min()),
                "max": float(pitch_vals.max()),
                "mean": float(pitch_vals.mean()),
                "median": float(pitch_vals.median()),
                "std": float(pitch_vals.std()),
            }

            pitch_dist = group["gongche"].value_counts().head(10).to_dict()
            # 归一化
            total = sum(pitch_dist.values())
            pitch_dist_norm = {str(k): round(v / total, 4) for k, v in pitch_dist.items()}

            # --- 骨架音 — 每个宫调的最常见音程序列 ---
            interval_dist = self._mode_interval_distribution(group)

            # --- 节奏特征 ---
            # 关联 beats_df
            mode_song_ids = group["polyu_id"].unique()
            mode_beats = beats_df[beats_df["polyu_id"].astype(str).isin(
                [str(x) for x in mode_song_ids]
            )]

            rubato_rate = 0.0
            beat_type_dist = {}
            avg_tone_yiyang = 0.0
            if len(mode_beats) > 0:
                rubato_rate = float(mode_beats["is_rubato"].mean())
                beat_type_dist = mode_beats["type"].value_counts().to_dict()
                beat_type_dist = {str(k): int(v) for k, v in beat_type_dist.items()}
                avg_tone_yiyang = float(mode_beats["tone_yiyang"].mean())

            # --- 南/北比例 ---
            region_dist = group["volume_region"].value_counts().to_dict()
            region_dist = {str(k): int(v) for k, v in region_dist.items() if pd.notna(k)}

            # --- 曲牌数 ---
            qupai_count = group["qupai_brackets_removed"].nunique()

            # --- 情感推断 (基于音域和节奏) ---
            emotional_hint = self._infer_emotional_quality(
                pitch_range, rubato_rate, avg_tone_yiyang,
                str(group["volume_region"].mode().iloc[0]) if len(group) > 0 else ""
            )

            features[mode] = {
                "song_count": int(group["polyu_id"].nunique()),
                "qupai_count": int(qupai_count),
                "total_notes": int(len(group)),
                "pitch_range": pitch_range,
                "pitch_distribution": pitch_dist_norm,
                "top_pitches": {str(k): int(v) for k, v in pitch_dist.items()},
                "interval_distribution": interval_dist,
                "rubato_rate": round(rubato_rate, 4),
                "beat_type_distribution": beat_type_dist,
                "avg_tone_yiyang": round(avg_tone_yiyang, 3),
                "region_distribution": region_dist,
                "emotional_hint": emotional_hint,
            }

        return features

    # ==================================================================
    # 4. 南词 vs 北词
    # ==================================================================

    def _extract_region_features(self, merged: pd.DataFrame) -> Dict:
        """量化南词与北词的风格差异"""
        features = {}

        for region in ["南詞", "北詞"]:
            subset = merged[merged["volume_region"] == region]
            if len(subset) == 0:
                continue

            # 音高分布
            pitch_dist = subset["gongche"].value_counts().head(10).to_dict()
            total = sum(pitch_dist.values())
            pitch_dist_norm = {str(k): round(v / total, 4) for k, v in pitch_dist.items()}

            # 音程分布（排除跨歌曲边界）
            sorted_subset = subset.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"])
            pitches = sorted_subset["gongche_pitch"].values
            pid_values = sorted_subset["polyu_id"].values
            intervals = np.diff(pitches)
            cross_mask = (pid_values[:-1] != pid_values[1:])
            interval_dist = self._safe_interval_distribution(intervals[~cross_mask])

            # 拖腔率
            melisma_rate = float((subset["multi_rhythm_num"] > 0).mean())

            # 音域
            pitch_vals = subset["gongche_pitch"].dropna()

            # 音符密度
            per_song = subset.groupby("polyu_id").agg(
                n_notes=("gongche", "count"),
                n_lyrics=("lyric_id", "nunique"),
            )
            per_song["density"] = per_song["n_notes"] / per_song["n_lyrics"].clip(lower=1)

            # 典型音程 — 级进(≤2) vs 跳进(>2) 比例
            step_ratio = sum(v for k, v in interval_dist.items()
                           if abs(float(k)) <= 2.0) / max(sum(interval_dist.values()), 1)

            features[region] = {
                "song_count": int(subset["polyu_id"].nunique()),
                "total_notes": int(len(subset)),
                "pitch_distribution": pitch_dist_norm,
                "interval_distribution": interval_dist,
                "step_ratio": round(step_ratio, 3),         # 级进比例
                "leap_ratio": round(1.0 - step_ratio, 3),   # 跳进比例
                "melisma_rate": round(melisma_rate, 3),
                "pitch_range": {
                    "min": float(pitch_vals.min()),
                    "max": float(pitch_vals.max()),
                    "mean": float(pitch_vals.mean()),
                    "std": float(pitch_vals.std()),
                },
                "avg_density": float(per_song["density"].mean()),
                "qupai_count": int(subset["qupai_brackets_removed"].nunique()),
            }

        return features

    # ==================================================================
    # 全局统计
    # ==================================================================

    def _extract_global_stats(self, merged: pd.DataFrame) -> Dict:
        """全局统计：整体音域、常用音程、拖腔比例等"""
        pitch_vals = merged["gongche_pitch"].dropna()
        sorted_all = merged.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"])
        pitches_all = sorted_all["gongche_pitch"].values
        pid_all = sorted_all["polyu_id"].values
        intervals = np.diff(pitches_all)
        cross_mask = (pid_all[:-1] != pid_all[1:])
        safe_intervals = intervals[~cross_mask]

        return {
            "total_songs": int(merged["polyu_id"].nunique()),
            "total_notes": int(len(merged)),
            "pitch_range": {
                "min": float(pitch_vals.min()),
                "max": float(pitch_vals.max()),
                "mean": float(pitch_vals.mean()),
                "std": float(pitch_vals.std()),
            },
            "overall_pitch_distribution": dict(
                merged["gongche"].value_counts().head(10).astype(int)
            ),
            "overall_interval_distribution": self._safe_interval_distribution(safe_intervals),
            "overall_melisma_rate": float((merged["multi_rhythm_num"] > 0).mean()),
            "overall_avg_duration": float(merged["duration"].mean()),
            "duration_distribution": {
                str(k): int(v) for k, v in merged["duration"].value_counts().items()
            },
        }

    # ==================================================================
    # 辅助方法
    # ==================================================================

    @staticmethod
    def _safe_interval_distribution(intervals: np.ndarray,
                                     bins: int = 20) -> Dict[str, float]:
        """安全的音程分布统计（过滤异常值）"""
        valid = intervals[(intervals > -15) & (intervals < 15)]
        if len(valid) == 0:
            return {}
        hist, edges = np.histogram(valid, bins=bins)
        result = {}
        for i in range(len(hist)):
            center = round(float((edges[i] + edges[i+1]) / 2), 1)
            result[str(center)] = int(hist[i])
        return result

    @staticmethod
    def _mode_interval_distribution(group: pd.DataFrame) -> Dict[str, float]:
        """宫调的音程分布（排除跨歌曲边界）"""
        sorted_g = group.sort_values(["polyu_id", "beat_id", "lyric_id", "gongche_id"])
        pitches = sorted_g["gongche_pitch"].values
        pid_values = sorted_g["polyu_id"].values
        intervals = np.diff(pitches)
        cross_mask = (pid_values[:-1] != pid_values[1:])
        valid = intervals[(~cross_mask) & (intervals > -15) & (intervals < 15)]
        if len(valid) == 0:
            return {}

        # 按大小分类
        step_up = int(((valid > 0) & (valid <= 2)).sum())       # 上行级进
        step_down = int(((valid < 0) & (valid >= -2)).sum())    # 下行级进
        leap_up = int(((valid > 2) & (valid <= 5)).sum())       # 上行小跳
        leap_down = int(((valid < -2) & (valid >= -5)).sum())   # 下行小跳
        big_up = int((valid > 5).sum())                          # 上行大跳
        big_down = int((valid < -5).sum())                       # 下行大跳
        same_pitch = int((valid == 0).sum())                     # 同音

        total = max(len(valid), 1)
        return {
            "step_up": round(step_up / total, 3),
            "step_down": round(step_down / total, 3),
            "leap_up": round(leap_up / total, 3),
            "leap_down": round(leap_down / total, 3),
            "big_jump_up": round(big_up / total, 3),
            "big_jump_down": round(big_down / total, 3),
            "same_pitch": round(same_pitch / total, 3),
        }

    @staticmethod
    def _infer_emotional_quality(pitch_range: dict, rubato_rate: float,
                                  avg_yiyang: float, region: str) -> str:
        """基于音域、散板率、依样度推断宫调的情感特质"""
        span = pitch_range["max"] - pitch_range["min"]

        hints = []
        if span > 12:
            hints.append("音域宽广")
        elif span > 8:
            hints.append("音域适中")
        else:
            hints.append("音域较窄")

        if rubato_rate > 0.1:
            hints.append("散板较多，自由舒缓")
        elif rubato_rate > 0.03:
            hints.append("偶有散板")
        else:
            hints.append("节奏规整")

        if avg_yiyang > 1.2:
            hints.append("旋律华彩装饰丰富")
        elif avg_yiyang > 0.8:
            hints.append("旋律装饰适中")
        else:
            hints.append("旋律简洁流畅")

        if "南" in region:
            hints.append("风格偏柔婉细腻")
        elif "北" in region:
            hints.append("风格偏刚健豪放")

        return "；".join(hints)

    # ==================================================================
    # 缓存
    # ==================================================================

    def _save_cache(self, result: Dict, cache_path: str):
        """保存为 JSON（可读）+ pickle（快速加载）"""
        path = Path(cache_path)

        # JSON 版本（人类可读）
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2,
                      default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x))
        logger.info(f"特征缓存(JSON): {json_path} ({json_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Pickle 版本（快速加载）
        pkl_path = path.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"特征缓存(PKL): {pkl_path} ({pkl_path.stat().st_size / 1024 / 1024:.1f} MB)")

    @staticmethod
    def load_cache(cache_path: str) -> Dict:
        """从缓存加载特征"""
        path = Path(cache_path)
        pkl_path = path.with_suffix(".pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
