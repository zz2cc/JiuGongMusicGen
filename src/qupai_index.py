"""
曲牌索引与 Few-Shot 检索模块。

对 2,176 个曲牌建立索引，支持：
1. 精确匹配：按曲牌名检索所有歌曲
2. 宫调匹配：按同宫调检索（备选）
3. 相似长度匹配：按歌词长度相似度排序
4. 音高骨架匹配：按曲牌旋律轮廓相似度检索
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .data_loader import JiuGongDataset, Song, NoteEvent
from .gongche_vocab import GONGCHE_TO_PITCH, melody_similarity, get_pitch_distribution

logger = logging.getLogger(__name__)


@dataclass
class QupaiProfile:
    """曲牌的统计画像"""
    qupai: str
    song_count: int
    polyu_ids: List[str] = field(default_factory=list)

    # 音高统计
    pitch_distribution: Dict[str, float] = field(default_factory=dict)
    avg_pitch_range: Tuple[float, float] = (0.0, 0.0)
    common_start_pitches: Dict[str, float] = field(default_factory=dict)
    common_end_pitches: Dict[str, float] = field(default_factory=dict)

    # 节奏统计
    avg_note_count: float = 0.0
    avg_beat_count: float = 0.0
    avg_notes_per_lyric: float = 0.0
    avg_melisma_ratio: float = 0.0

    # 来源与宫调
    sources: Dict[str, int] = field(default_factory=dict)
    modes: Dict[str, int] = field(default_factory=dict)
    regions: Dict[str, int] = field(default_factory=dict)

    # 代表性旋律序列（用于骨架匹配）
    skeleton_pitches: List[float] = field(default_factory=list)


class QupaiIndex:
    """
    曲牌索引，提供快速检索和 Few-Shot 样例选择。

    Usage:
        index = QupaiIndex(dataset)
        index.build()
        examples = index.retrieve_examples("奉時春", lyrics="風和日麗", n=3)
    """

    def __init__(self, dataset: JiuGongDataset):
        self.dataset = dataset
        self.profiles: Dict[str, QupaiProfile] = {}
        self._built = False

        # 宫调→曲牌反向索引
        self._mode_to_qupai: Dict[str, List[str]] = defaultdict(list)

    def build(self, min_song_count: int = 1):
        """
        构建所有曲牌的统计画像。

        Args:
            min_song_count: 最少歌曲数阈值（低于此值只建基本索引）
        """
        logger.info("构建曲牌索引...")
        qupai_groups: Dict[str, List[str]] = defaultdict(list)

        # 分组
        for pid in self.dataset.get_all_polyu_ids():
            try:
                song = self.dataset.get_song(pid)
                qupai_groups[song.qupai_bare].append(pid)
            except KeyError:
                continue

        for qupai, pids in qupai_groups.items():
            if len(pids) < min_song_count:
                continue

            profile = self._build_profile(qupai, pids)
            self.profiles[qupai] = profile

            # 宫调索引
            for mode in profile.modes:
                self._mode_to_qupai[mode].append(qupai)

        self._built = True
        logger.info(f"曲牌索引构建完成: {len(self.profiles)} 个曲牌")

    def _build_profile(self, qupai: str, pids: List[str]) -> QupaiProfile:
        """构建单个曲牌的统计画像"""
        profile = QupaiProfile(qupai=qupai, song_count=len(pids), polyu_ids=pids)

        all_notes = []
        all_pitches = []
        start_pitches = []
        end_pitches = []
        note_counts = []
        beat_counts = []
        notes_per_lyric = []
        melisma_ratios = []
        sources = defaultdict(int)
        modes = defaultdict(int)
        regions = defaultdict(int)

        for pid in pids:
            try:
                song = self.dataset.get_song(pid)
                all_notes.extend(song.notes)

                pitches = [n.gongche_pitch for n in song.notes]
                all_pitches.extend(pitches)

                if song.notes:
                    start_pitches.append(song.notes[0].gongche)
                    end_pitches.append(song.notes[-1].gongche)

                note_counts.append(song.note_count)
                beat_counts.append(song.beat_count)
                notes_per_lyric.append(song.notes_per_lyric)
                melisma_ratios.append(song.melisma_ratio)

                sources[song.source] += 1
                modes[song.volume_mode] += 1
                regions[song.volume_region] += 1
            except KeyError:
                continue

        # 音高分布
        from collections import Counter
        if all_pitches:
            pitch_counter = Counter(n.gongche for n in all_notes)
            total = len(all_notes)
            profile.pitch_distribution = {gc: c / total
                                          for gc, c in pitch_counter.items()}

            profile.avg_pitch_range = (
                min(all_pitches) if all_pitches else 0.0,
                max(all_pitches) if all_pitches else 0.0,
            )

        if start_pitches:
            sc = Counter(start_pitches)
            total_s = len(start_pitches)
            profile.common_start_pitches = {gc: c / total_s
                                            for gc, c in sc.most_common(5)}

        if end_pitches:
            ec = Counter(end_pitches)
            total_e = len(end_pitches)
            profile.common_end_pitches = {gc: c / total_e
                                          for gc, c in ec.most_common(5)}

        # 统计量
        profile.avg_note_count = np.mean(note_counts) if note_counts else 0
        profile.avg_beat_count = np.mean(beat_counts) if beat_counts else 0
        profile.avg_notes_per_lyric = np.mean(notes_per_lyric) if notes_per_lyric else 0
        profile.avg_melisma_ratio = np.mean(melisma_ratios) if melisma_ratios else 0
        profile.sources = dict(sources)
        profile.modes = dict(modes)
        profile.regions = dict(regions)

        # 骨架音高序列（取所有歌曲的平均音高轮廓）
        if note_counts:
            median_len = int(np.median(note_counts))
            if median_len > 0 and len(pids) >= 3:
                # 取所有歌曲中第一首的完整音高做骨架
                try:
                    first_song = self.dataset.get_song(pids[0])
                    profile.skeleton_pitches = [n.gongche_pitch
                                                for n in first_song.notes]
                except KeyError:
                    pass

        return profile

    def get_profile(self, qupai: str) -> Optional[QupaiProfile]:
        """获取曲牌画像"""
        qupai_bare = qupai.replace("《", "").replace("》", "")
        return self.profiles.get(qupai_bare)

    def has_qupai(self, qupai: str) -> bool:
        """检查曲牌是否存在"""
        qupai_bare = qupai.replace("《", "").replace("》", "")
        return qupai_bare in self.profiles

    def get_polyu_ids(self, qupai: str) -> List[str]:
        """获取某曲牌的所有歌曲ID"""
        profile = self.get_profile(qupai)
        if profile:
            return profile.polyu_ids
        # 回退到直接查询
        return self.dataset._qupai_index.get(
            qupai.replace("《", "").replace("》", ""), []
        )

    def retrieve_examples(
        self,
        qupai: str,
        lyrics: str = "",
        n: int = 3,
        prefer_same_qupai: bool = True,
    ) -> List[Dict]:
        """
        检索 Few-Shot 样例。

        策略：
        1. 优先同曲牌、相似歌词长度的歌曲
        2. 不足时从同宫调曲牌补充
        3. 最终不足时随机选取

        Args:
            qupai: 曲牌名
            lyrics: 目标歌词（用于长度匹配）
            n: 样例数量
            prefer_same_qupai: 是否优先同曲牌

        Returns:
            [{"polyu_id": str, "qupai": str, "lyrics": str,
              "gongche_text": str, "score": float}, ...]
        """
        target_len = len(lyrics) if lyrics else 0
        candidates = []

        # ---- 1. 同曲牌 ----
        qupai_bare = qupai.replace("《", "").replace("》", "")
        same_qupai_ids = self.get_polyu_ids(qupai_bare)

        if same_qupai_ids and prefer_same_qupai:
            for pid in same_qupai_ids:
                try:
                    song = self.dataset.get_song(pid)
                    lyric_len = len(song.lyrics_text) if song.lyrics_text else 0
                    length_score = 1.0 / (1.0 + abs(lyric_len - target_len) * 0.1)
                    candidates.append((song, length_score))
                except KeyError:
                    continue

        if candidates:
            # 按长度相似度 + 去重排序
            candidates.sort(key=lambda x: -x[1])
            seen_texts = set()
            deduped = []
            for song, score in candidates:
                text = song.lyrics_text
                if text not in seen_texts and len(deduped) < n:
                    seen_texts.add(text)
                    deduped.append((song, score))
            candidates = deduped

        # ---- 2. 同宫调补充 ----
        if len(candidates) < n:
            profile = self.get_profile(qupai_bare)
            if profile and profile.modes:
                main_mode = max(profile.modes, key=profile.modes.get)
                same_mode_qupais = self._mode_to_qupai.get(main_mode, [])
                for sq in same_mode_qupais:
                    if sq == qupai_bare:
                        continue
                    for pid in self.get_polyu_ids(sq):
                        if len(candidates) >= n * 2:
                            break
                        try:
                            song = self.dataset.get_song(pid)
                            lyric_len = len(song.lyrics_text) if song.lyrics_text else 0
                            score = 0.5 / (1.0 + abs(lyric_len - target_len) * 0.1)
                            candidates.append((song, score))
                        except KeyError:
                            continue

        # ---- 去重 & 排序 & 截断 ----
        seen_texts = set()
        final = []
        candidates.sort(key=lambda x: -x[1])
        for song, score in candidates:
            text = song.lyrics_text
            if text and text not in seen_texts:
                seen_texts.add(text)
                final.append({
                    "polyu_id": song.polyu_id,
                    "qupai": song.qupai_bare,
                    "lyrics": text,
                    "gongche_text": song.gongche_text,
                    "lyric_note_groups": song.lyric_note_groups,
                    "notes": song.notes,
                    "source": song.source,
                    "volume_region": song.volume_region,
                    "volume_mode": song.volume_mode,
                    "note_count": song.note_count,
                    "beat_count": song.beat_count,
                    "notes_per_lyric": song.notes_per_lyric,
                    "score": score,
                })
            if len(final) >= n:
                break

        return final

    def search_by_lyrics(self, text: str, top_k: int = 5) -> List[Dict]:
        """按歌词片段搜索相似歌曲"""
        results = []
        polyu_ids = self.dataset.search_lyrics(text)
        for pid in polyu_ids[:top_k * 2]:
            try:
                song = self.dataset.get_song(pid)
                results.append({
                    "polyu_id": song.polyu_id,
                    "qupai": song.qupai_bare,
                    "lyrics": song.lyrics_text,
                    "gongche_text": song.gongche_text,
                    "source": song.source,
                })
            except KeyError:
                continue

        return results[:top_k]

    def get_qupai_by_mode(self, mode: str, top_k: int = 10) -> List[str]:
        """获取某宫调下的曲牌列表（按歌曲数排序）"""
        qupais = self._mode_to_qupai.get(mode, [])
        qupais.sort(key=lambda q: self.profiles.get(q, QupaiProfile(qupai=q, song_count=0)).song_count,
                     reverse=True)
        return qupais[:top_k]

    def get_similar_qupai(self, qupai: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        基于音高分布找相似曲牌。

        Returns:
            [(qupai_name, similarity_score), ...]
        """
        profile = self.get_profile(qupai)
        if not profile or not profile.pitch_distribution:
            return []

        scores = []
        for qname, qprof in self.profiles.items():
            if qname == qupai:
                continue
            if not qprof.pitch_distribution:
                continue

            # JS 散度近似
            all_gc = set(profile.pitch_distribution) | set(qprof.pitch_distribution)
            js = 0.0
            for gc in all_gc:
                p = profile.pitch_distribution.get(gc, 0.0)
                q = qprof.pitch_distribution.get(gc, 0.0)
                m = (p + q) / 2
                if m > 0:
                    if p > 0:
                        js += p * np.log(p / m) / 2
                    if q > 0:
                        js += q * np.log(q / m) / 2

            similarity = 1.0 / (1.0 + np.sqrt(abs(js)))
            scores.append((qname, similarity))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self.profiles)
