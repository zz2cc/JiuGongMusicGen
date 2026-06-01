"""
数据加载模块 — 从CSV文件加载九宫大成数据集。
提供歌曲元数据、音符级数据、板拍级数据的查询接口。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import pickle
import logging

from .config import SONGS_CSV, NOTES_CSV, BEATS_CSV, ROOT_DIR

logger = logging.getLogger(__name__)

# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class NoteEvent:
    """单个音符事件 — 对应 FINAL_NOTES.csv 中的一行"""
    lyric: str              # 歌词字 "日"
    lyric_modern_tone: int  # 现代声调 0-4
    lyric_is_entering: bool # 是否入声字
    lyric_guangyun_tone: str  # 广韵声调 平/上/去/入
    gongche: str            # 工尺谱字符 "上"
    gongche_pitch: float    # 音高数值 0.0
    duration: float         # 时长 (四分音符比例)
    beat_id: int            # 所属板拍ID
    line_id: int            # 行号
    lyric_id: int           # 歌词字序号
    is_melisma: bool        # 是否为拖腔后续音 (multi_rhythm_num > 0)
    melisma_id: int         # 拖腔组ID
    rhythm: str             # 节奏标记 (。=板, 、=眼, etc.)


@dataclass
class LyricNoteGroup:
    """一个歌词字对应的所有音符（一字多音/拖腔）"""
    lyric: str
    lyric_modern_tone: int
    lyric_is_entering: bool
    lyric_guangyun_tone: str
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def gongche_sequence(self) -> List[str]:
        return [n.gongche for n in self.notes]

    @property
    def pitch_sequence(self) -> List[float]:
        return [n.gongche_pitch for n in self.notes]

    @property
    def duration_sequence(self) -> List[float]:
        return [n.duration for n in self.notes]


@dataclass
class Song:
    """一首歌曲的完整数据"""
    polyu_id: str           # 歌曲ID "287.1"
    note_count: int         # 总音符数
    beat_count: int         # 总板拍数
    qupai: str              # 曲牌名 "奉時春"
    qupai_bare: str         # 曲牌名(无书名号) "奉時春"
    source: str             # 来源 "月令承應"
    volume: str             # 卷号
    volume_region: str      # 南词/北词/合套
    volume_mode: str        # 宫调 "仙呂宮引"
    is_duration_pred: bool  # 节奏是否推算
    has_rhythm_marked: bool # 原始是否标板眼
    lyric_note_groups: List[LyricNoteGroup] = field(default_factory=list)
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def lyrics_text(self) -> str:
        """获取完整歌词文本"""
        return "".join(g.lyric for g in self.lyric_note_groups)

    @property
    def gongche_text(self) -> str:
        """获取工尺谱完整序列"""
        return " ".join(n.gongche for n in self.notes)

    @property
    def melisma_ratio(self) -> float:
        """拖腔比例 = 拖腔音符数 / 总音符数"""
        if not self.notes:
            return 0.0
        melisma_count = sum(1 for n in self.notes if n.is_melisma)
        return melisma_count / len(self.notes)

    @property
    def notes_per_lyric(self) -> float:
        """平均每字音符数"""
        if not self.lyric_note_groups:
            return 0.0
        return len(self.notes) / len(self.lyric_note_groups)


# ============================================================================
# 数据集加载器
# ============================================================================


class JiuGongDataset:
    """
    九宫大成数据集加载器。

    Usage:
        ds = JiuGongDataset()
        ds.load()
        songs = ds.get_songs_by_qupai("奉時春")
        song = ds.get_song("287.1")
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = ROOT_DIR
        self.data_dir = Path(data_dir)
        self.songs_csv = self.data_dir / "FINAL_SONGS.csv"
        self.notes_csv = self.data_dir / "FINAL_NOTES.csv"
        self.beats_csv = self.data_dir / "FINAL_BEATS.csv"

        # DataFrame 缓存
        self._songs_df: Optional[pd.DataFrame] = None
        self._notes_df: Optional[pd.DataFrame] = None
        self._beats_df: Optional[pd.DataFrame] = None

        # 索引缓存
        self._songs_cache: Dict[str, Song] = {}
        self._qupai_index: Dict[str, List[str]] = defaultdict(list)
        self._mode_index: Dict[str, List[str]] = defaultdict(list)
        self._region_index: Dict[str, List[str]] = defaultdict(list)

        self._loaded = False

    # ---- 加载 ----

    def load(self, cache_path: Optional[str] = None) -> "JiuGongDataset":
        """加载所有CSV数据并建立索引。支持 pickle 缓存加速。"""
        if cache_path and Path(cache_path).exists():
            logger.info(f"从缓存加载: {cache_path}")
            return self._load_from_cache(cache_path)

        logger.info("加载 FINAL_SONGS.csv ...")
        self._songs_df = pd.read_csv(self.songs_csv, sep=",", encoding="utf-8")
        # 用第一行修正列名（FINAL_SONGS 第一行是编号行）
        if "polyu_id" not in self._songs_df.columns:
            # 第一列可能被读取为索引
            cols = self._songs_df.columns.tolist()
            logger.info(f"SONGS 列: {cols}")

        logger.info(f"  已加载 {len(self._songs_df)} 首歌曲")

        logger.info("加载 FINAL_NOTES.csv ...")
        self._notes_df = pd.read_csv(self.notes_csv, sep=",", encoding="utf-8", low_memory=False)
        logger.info(f"  已加载 {len(self._notes_df)} 行音符数据")

        logger.info("加载 FINAL_BEATS.csv ...")
        self._beats_df = pd.read_csv(self.beats_csv, sep=",", encoding="utf-8", low_memory=False)
        logger.info(f"  已加载 {len(self._beats_df)} 行板拍数据")

        logger.info("建立索引 ...")
        self._build_indices()

        self._loaded = True
        logger.info("数据集加载完成。")

        if cache_path:
            self._save_to_cache(cache_path)

        return self

    def _load_from_cache(self, cache_path: str) -> "JiuGongDataset":
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self._songs_df = data["songs_df"]
        self._notes_df = data["notes_df"]
        self._beats_df = data["beats_df"]
        self._qupai_index = data["qupai_index"]
        self._mode_index = data["mode_index"]
        self._region_index = data["region_index"]
        self._loaded = True
        logger.info("缓存加载完成。")
        return self

    def _save_to_cache(self, cache_path: str):
        data = {
            "songs_df": self._songs_df,
            "notes_df": self._notes_df,
            "beats_df": self._beats_df,
            "qupai_index": dict(self._qupai_index),
            "mode_index": dict(self._mode_index),
            "region_index": dict(self._region_index),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"缓存已保存: {cache_path}")

    def _build_indices(self):
        """建立曲牌、宫调、板块的反向索引"""
        songs_df = self._songs_df

        for _, row in songs_df.iterrows():
            pid = str(row["polyu_id"])

            # 曲牌索引
            qupai_bare = str(row.get("qupai_brackets_removed", ""))
            qupai_full = str(row.get("qupai", ""))
            self._qupai_index[qupai_bare].append(pid)
            if qupai_full and qupai_full != qupai_bare:
                self._qupai_index[qupai_full].append(pid)

            # 宫调索引
            mode = str(row.get("volume_mode", ""))
            if mode and mode != "nan":
                self._mode_index[mode].append(pid)

            # 板块索引
            region = str(row.get("volume_region", ""))
            if region and region != "nan":
                self._region_index[region].append(pid)

    # ---- 查询接口 ----

    @property
    def songs_df(self) -> pd.DataFrame:
        if self._songs_df is None:
            raise RuntimeError("请先调用 .load()")
        return self._songs_df

    @property
    def notes_df(self) -> pd.DataFrame:
        if self._notes_df is None:
            raise RuntimeError("请先调用 .load()")
        return self._notes_df

    @property
    def beats_df(self) -> pd.DataFrame:
        if self._beats_df is None:
            raise RuntimeError("请先调用 .load()")
        return self._beats_df

    def get_all_polyu_ids(self) -> List[str]:
        """获取所有歌曲ID"""
        return [str(pid) for pid in self.songs_df["polyu_id"].tolist()]

    def get_songs_by_qupai(self, qupai: str) -> List[Song]:
        """按曲牌名查询所有歌曲"""
        # 去掉书名号
        qupai_bare = qupai.replace("《", "").replace("》", "")
        ids = self._qupai_index.get(qupai_bare, [])
        if not ids:
            # 模糊匹配
            ids = self._fuzzy_match_qupai(qupai_bare)
        return [self.get_song(pid) for pid in ids]

    def get_songs_by_region(self, region: str) -> List[Song]:
        """按板块查询（南词/北词/合套）"""
        ids = self._region_index.get(region, [])
        return [self.get_song(pid) for pid in ids]

    def get_songs_by_mode(self, mode: str) -> List[Song]:
        """按宫调查询"""
        ids = self._mode_index.get(mode, [])
        return [self.get_song(pid) for pid in ids]

    def get_song(self, polyu_id: str) -> Song:
        """获取单首歌曲的完整数据"""
        pid = str(polyu_id)

        # 检查缓存
        if pid in self._songs_cache:
            return self._songs_cache[pid]

        # 查元数据
        songs_df = self.songs_df
        song_rows = songs_df[songs_df["polyu_id"].astype(str) == pid]
        if len(song_rows) == 0:
            raise KeyError(f"歌曲 {pid} 不存在")

        row = song_rows.iloc[0]
        song = Song(
            polyu_id=pid,
            note_count=int(row.get("note_count", 0)),
            beat_count=int(row.get("beat_count", 0)),
            qupai=str(row.get("qupai", "")),
            qupai_bare=str(row.get("qupai_brackets_removed", "")),
            source=str(row.get("source", "")),
            volume=str(row.get("volume", "")),
            volume_region=str(row.get("volume_region", "")),
            volume_mode=str(row.get("volume_mode", "")),
            is_duration_pred=str(row.get("is_duration_pred", "")).lower() == "true",
            has_rhythm_marked=str(row.get("has_rhythm_marked", "")).lower() == "true",
        )

        # 查音符数据
        notes_df = self.notes_df
        note_rows = notes_df[notes_df["polyu_id"].astype(str) == pid]
        note_rows = note_rows.sort_values(["beat_id", "lyric_id", "gongche_id"])

        notes = []
        for _, nr in note_rows.iterrows():
            multi_num = int(nr.get("multi_rhythm_num", 0))
            note = NoteEvent(
                lyric=str(nr.get("lyric", "")),
                lyric_modern_tone=int(nr.get("lyric_modern_tone", 0)),
                lyric_is_entering=str(nr.get("lyric_is_entering", "")).lower() == "true",
                lyric_guangyun_tone=str(nr.get("lyric_guangyun_tone", "")),
                gongche=str(nr.get("gongche", "")),
                gongche_pitch=float(nr.get("gongche_pitch", 0)),
                duration=float(nr.get("duration", 0.25)),
                beat_id=int(nr.get("beat_id", 0)),
                line_id=int(nr.get("line_id", 0)),
                lyric_id=int(nr.get("lyric_id", 0)),
                is_melisma=(multi_num > 0),
                melisma_id=int(nr.get("multi_rhythm_id", 0)),
                rhythm=str(nr.get("rhythm", "")),
            )
            notes.append(note)

        song.notes = notes

        # 构建歌词-音符分组（按 lyric_id 聚合）
        groups: Dict[Tuple[int, int], List[NoteEvent]] = defaultdict(list)
        for n in notes:
            key = (n.line_id, n.lyric_id)
            groups[key].append(n)

        lyric_groups = []
        for key in sorted(groups.keys()):
            gn = groups[key]
            first = gn[0]
            lyric_groups.append(LyricNoteGroup(
                lyric=first.lyric,
                lyric_modern_tone=first.lyric_modern_tone,
                lyric_is_entering=first.lyric_is_entering,
                lyric_guangyun_tone=first.lyric_guangyun_tone,
                notes=sorted(gn, key=lambda x: x.beat_id * 1000 + x.melisma_id),
            ))

        song.lyric_note_groups = lyric_groups

        # 缓存
        self._songs_cache[pid] = song
        return song

    def _fuzzy_match_qupai(self, qupai_bare: str) -> List[str]:
        """模糊匹配曲牌名"""
        results = []
        for qname, ids in self._qupai_index.items():
            if qupai_bare in qname or qname in qupai_bare:
                results.extend(ids)
        return list(set(results)) if results else []

    # ---- 统计接口 ----

    def get_qupai_list(self, min_count: int = 1) -> pd.DataFrame:
        """获取曲牌列表及出现次数"""
        songs_df = self.songs_df
        qupai_counts = songs_df["qupai_brackets_removed"].value_counts()
        qupai_counts = qupai_counts[qupai_counts >= min_count]
        result = qupai_counts.reset_index()
        result.columns = ["qupai", "count"]
        return result

    def get_mode_list(self) -> Dict[str, int]:
        """获取宫调分布"""
        return dict(self.songs_df["volume_mode"].value_counts())

    def get_region_distribution(self) -> Dict[str, int]:
        """获取南词/北词/合套分布"""
        return dict(self.songs_df["volume_region"].value_counts())

    def get_qupai_pitch_profile(self, qupai: str) -> Dict[str, any]:
        """
        获取某曲牌的典型音高统计信息。

        Returns:
            {
                "pitch_distribution": {gongche: count, ...},
                "starting_pitches": {gongche: count, ...},   # 起音分布
                "ending_pitches": {gongche: count, ...},     # 收音分布
                "pitch_range": (min_pitch, max_pitch),
                "avg_notes_per_lyric": float,
                "avg_duration_per_note": float,
                "melisma_ratio": float,
            }
        """
        songs = self.get_songs_by_qupai(qupai)
        if not songs:
            return {}

        all_pitches = []
        start_pitches = []
        end_pitches = []
        note_counts = []
        melisma_ratios = []

        for song in songs:
            if song.notes:
                all_pitches.extend(n.gongche for n in song.notes)
                start_pitches.append(song.notes[0].gongche)
                end_pitches.append(song.notes[-1].gongche)
                note_counts.append(len(song.notes))
                melisma_ratios.append(song.melisma_ratio)

        from collections import Counter
        return {
            "pitch_distribution": dict(Counter(all_pitches)),
            "starting_pitches": dict(Counter(start_pitches)),
            "ending_pitches": dict(Counter(end_pitches)),
            "pitch_range": (
                min(n.gongche_pitch for s in songs for n in s.notes),
                max(n.gongche_pitch for s in songs for n in s.notes),
            ),
            "avg_notes_per_lyric": np.mean([s.notes_per_lyric for s in songs]) if songs else 0,
            "melisma_ratio": np.mean(melisma_ratios) if melisma_ratios else 0,
        }

    def search_lyrics(self, text: str, qupai: Optional[str] = None) -> List[str]:
        """按歌词文本搜索歌曲ID"""
        notes_df = self.notes_df
        matching = notes_df[notes_df["lyric"].astype(str).str.contains(text, na=False)]
        ids = matching["polyu_id"].unique().tolist()
        if qupai:
            qupai_ids = set(self._qupai_index.get(qupai, []))
            ids = [i for i in ids if i in qupai_ids]
        return ids

    # ---- 批量操作 ----

    def iter_songs(self, polyu_ids: Optional[List[str]] = None,
                   batch_size: int = 100) -> "Iterator[List[Song]]":
        """批量迭代歌曲，减少内存压力"""
        if polyu_ids is None:
            polyu_ids = self.get_all_polyu_ids()

        for i in range(0, len(polyu_ids), batch_size):
            batch = polyu_ids[i:i + batch_size]
            yield [self.get_song(pid) for pid in batch]

    def __len__(self) -> int:
        return len(self.songs_df)


# ============================================================================
# 便捷函数
# ============================================================================

# 全局单例（懒加载）
_global_dataset: Optional[JiuGongDataset] = None


def get_dataset(data_dir: Optional[Path] = None,
                cache_path: Optional[str] = None) -> JiuGongDataset:
    """获取全局数据集实例（单例模式）"""
    global _global_dataset
    if _global_dataset is not None and _global_dataset._loaded:
        if data_dir is not None and data_dir != _global_dataset.data_dir:
            logger.warning(
                f"get_dataset(data_dir={data_dir}) 被忽略——已加载单例 "
                f"(data_dir={_global_dataset.data_dir})。如需切换数据集，"
                f"请先调用 JiuGongDataset(data_dir=...).load() 创建新实例。"
            )
        return _global_dataset
    _global_dataset = JiuGongDataset(data_dir=data_dir)
    _global_dataset.load(cache_path=cache_path)
    return _global_dataset
