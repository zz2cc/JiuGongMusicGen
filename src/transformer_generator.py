"""
条件式 Transformer 旋律生成器（任务三模型集成）。

将任务三的 ConditionalMelodyTransformer 模型输出（MIDI+时长）
转换为任务四的 LyricNoteGroup，从而复用 MusicXML 和评估系统。
"""

import os
import re
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhconv import convert
from xpinyin import Pinyin

from .gongche_vocab import (
    GONGCHE_TO_PITCH, MIDI_TO_GONGCHE, GONGCHE_TO_MIDI,
    GONGCHE_CHARS,
)
from .data_loader import NoteEvent, LyricNoteGroup

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2
PUNCT_COMMA = "，,、"
PUNCT_PERIOD = "。.！？!?；;"

_py = Pinyin()


def _get_tone(c: str) -> int:
    """取单字声调 1-4，取不到记 0。与训练时一致。"""
    p = _py.get_pinyin(convert(c, "zh-cn"), tone_marks="numbers")
    m = re.search(r"[1-4]", p)
    return int(m.group()) if m else 0


def _midi_to_gongche(midi: int) -> Tuple[str, float]:
    """MIDI 音高 → 最近工尺字符 + 音高值。

    Returns:
        (gongche_char, pitch_value)
    """
    if midi in MIDI_TO_GONGCHE:
        gc = MIDI_TO_GONGCHE[midi]
        return gc, GONGCHE_TO_PITCH[gc]

    # 最近邻查找
    best_gc = "上"
    best_dist = 999
    for gc, gc_midi in GONGCHE_TO_MIDI.items():
        dist = abs(midi - gc_midi)
        if dist < best_dist:
            best_dist = dist
            best_gc = gc
    return best_gc, GONGCHE_TO_PITCH[best_gc]


def _dur_to_quarter(dur_value: float) -> float:
    """模型输出的时长(dur_vocab) → 四分音符比例 """
    return dur_value


# ---------------------------------------------------------------------------
# 模型结构（与任务三 app.py 完全一致）
# ---------------------------------------------------------------------------
class ConditionalMelodyTransformer(nn.Module):
    def __init__(
        self,
        num_pitches,
        num_durs,
        num_gongs,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.15,
        max_len=512,
    ):
        super().__init__()
        self.pitch_emb = nn.Embedding(num_pitches, d_model, padding_idx=PAD)
        self.dur_emb = nn.Embedding(num_durs, d_model, padding_idx=PAD)
        self.tone_emb = nn.Embedding(5, d_model)
        self.gong_emb = nn.Embedding(num_gongs, d_model)
        self.word_start_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.pitch_head = nn.Linear(d_model, num_pitches)
        self.dur_head = nn.Linear(d_model, num_durs)
        self.char_end_head = nn.Linear(d_model, 1)

    def forward(self, pitch_in, dur_in, tone_cond, word_start_cond, gong_id):
        B, T = pitch_in.shape
        pos = torch.arange(T, device=pitch_in.device).unsqueeze(0).expand(B, T)
        gong = self.gong_emb(gong_id).unsqueeze(1).expand(B, T, -1)
        h = (
            self.pitch_emb(pitch_in) + self.dur_emb(dur_in)
            + self.tone_emb(tone_cond) + self.word_start_emb(word_start_cond)
            + gong + self.pos_emb(pos)
        )
        causal_mask = torch.triu(
            torch.ones(T, T, device=pitch_in.device, dtype=torch.bool), diagonal=1,
        )
        pad_mask = pitch_in.eq(PAD)
        z = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        return (
            self.pitch_head(z),
            self.dur_head(z),
            self.char_end_head(z).squeeze(-1),
        )


# ---------------------------------------------------------------------------
# TransformerGenerator — 单例包装
# ---------------------------------------------------------------------------
class TransformerGenerator:
    """任务三条件式 Transformer 生成器。

    Usage:
        gen = TransformerGenerator("任务三交接/任务三/conditional_transformer_best.pt")
        groups = gen.generate("春风花草香", "仙呂宮引")
        # groups is List[LyricNoteGroup]，可直接传给 MusicXML + 评估
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.pitch_vocab = ckpt["pitch_vocab"]
        self.dur_vocab = ckpt["dur_vocab"]
        self.gongs: List[str] = ckpt["gongs"]
        self.gong2id: Dict[str, int] = ckpt["gong2id"]
        self.pitch2id: Dict[int, int] = ckpt["pitch2id"]
        self.dur2id: Dict[float, int] = ckpt["dur2id"]

        self.model = ConditionalMelodyTransformer(
            num_pitches=len(self.pitch_vocab),
            num_durs=len(self.dur_vocab),
            num_gongs=len(self.gongs),
            d_model=192, nhead=8, num_layers=4, dropout=0.15,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Transformer] 模型加载完毕 | 参数 {n_params:,} | 宫调 {len(self.gongs)} 个")

    def _top_p_sample(self, logits, top_p=0.9, temperature=0.85, banned_ids=None):
        logits = logits / max(temperature, 0.1)
        if banned_ids is not None:
            logits = logits.clone()
            for bid in banned_ids:
                logits[bid] = -1e9
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        filtered = filtered / filtered.sum().clamp(min=1e-8)
        return torch.multinomial(filtered, 1).item()

    def generate(
        self,
        lyrics: str,
        gongdiao: str,
        seed: int = 42,
        temperature: float = 0.85,
        top_p: float = 0.9,
        max_notes_per_char: int = 4,
    ) -> List[LyricNoteGroup]:
        """生成旋律，返回 LyricNoteGroup 列表（可直接传 MusicXML + 评估）。

        Args:
            lyrics: 歌词文本（可含标点）
            gongdiao: 宫调名（如 "仙呂宮引"）
            seed: 随机种子
            temperature: 采样温度
            top_p: nucleus sampling

        Returns:
            List[LyricNoteGroup]
        """
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        self.model.eval()

        if gongdiao not in self.gong2id:
            gongdiao = self.gongs[0]

        gong_id = torch.tensor([self.gong2id[gongdiao]], dtype=torch.long, device=self.device)

        pitch_ids = [BOS]
        dur_ids = [BOS]
        hist_tones = []
        hist_word_starts = []

        result_groups: List[LyricNoteGroup] = []
        lyric_id = 0

        for ch in lyrics:
            if ch in PUNCT_COMMA or ch in PUNCT_PERIOD:
                # 标点 → 跳过（MusicXML 会通过词间分隔自然处理）
                continue
            if not ch.strip():
                continue

            tone = _get_tone(ch)
            if tone not in [0, 1, 2, 3, 4]:
                tone = 0

            notes_raw = []

            for k in range(max_notes_per_char):
                pitch_in = torch.tensor([pitch_ids], dtype=torch.long, device=self.device)
                dur_in = torch.tensor([dur_ids], dtype=torch.long, device=self.device)
                word_start = 1 if k == 0 else 0

                tone_cond = torch.tensor(
                    [hist_tones + [tone]], dtype=torch.long, device=self.device,
                )
                ws_cond = torch.tensor(
                    [hist_word_starts + [word_start]], dtype=torch.long, device=self.device,
                )

                with torch.no_grad():
                    p_logits, d_logits, e_logits = self.model(
                        pitch_in, dur_in, tone_cond, ws_cond, gong_id,
                    )

                next_p = self._top_p_sample(
                    p_logits[0, -1], top_p=top_p, temperature=temperature,
                    banned_ids=[PAD, BOS],
                )
                next_d = self._top_p_sample(
                    d_logits[0, -1], top_p=top_p, temperature=temperature,
                    banned_ids=[PAD, BOS],
                )

                if next_p == EOS or next_d == EOS:
                    if len(notes_raw) == 0:
                        continue
                    break

                midi = self.pitch_vocab[next_p]
                dur = self.dur_vocab[next_d]
                notes_raw.append((int(midi), float(dur)))

                pitch_ids.append(next_p)
                dur_ids.append(next_d)
                hist_tones.append(tone)
                hist_word_starts.append(word_start)

                end_prob = torch.sigmoid(e_logits[0, -1]).item()
                if k >= 1 and end_prob > 0.5:
                    break

            if len(notes_raw) == 0:
                # 兜底：给一个默认音
                notes_raw = [(60, 1.0)]

            # 转为 NoteEvent 列表
            lyric_id += 1
            note_events = []
            for j, (midi, dur_q) in enumerate(notes_raw):
                gc, pitch_val = _midi_to_gongche(midi)
                ev = NoteEvent(
                    lyric=ch if j == 0 else "",
                    lyric_modern_tone=tone,
                    lyric_is_entering=False,
                    lyric_guangyun_tone="",
                    gongche=gc,
                    gongche_pitch=pitch_val,
                    duration=dur_q,          # ← 模型输出的时长，≈ 四分音符比例
                    beat_id=lyric_id,
                    line_id=1,
                    lyric_id=lyric_id,
                    is_melisma=(j > 0),
                    melisma_id=lyric_id if j > 0 else 0,
                    rhythm="",
                )
                note_events.append(ev)

            group = LyricNoteGroup(
                lyric=ch,
                lyric_modern_tone=tone,
                lyric_is_entering=False,
                lyric_guangyun_tone="",
                notes=note_events,
            )
            result_groups.append(group)

        return result_groups


# 模块级便捷函数（供 web_app 调用）

def load_generator(ckpt_path: str = None) -> TransformerGenerator:
    """加载 Transformer 生成器（单例）。

    Args:
        ckpt_path: .pt 文件路径，默认在 任务三交接/任务三/ 下
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "任务三交接", "任务三", "conditional_transformer_best.pt",
        )
    return TransformerGenerator(ckpt_path)


def list_gongs(ckpt_path: str = None) -> List[str]:
    """返回可用的宫调列表（不加载完整模型，只读 checkpoint）。"""
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "任务三交接", "任务三", "conditional_transformer_best.pt",
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["gongs"]
