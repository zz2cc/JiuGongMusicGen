"""
九宫旋律生成 · 后端服务
============================
输入歌词 + 选择宫调 -> 用条件式 Transformer 生成旋律 -> 返回可下载的 MusicXML。

运行：
    pip install fastapi uvicorn torch music21 zhconv xpinyin
    python app.py
然后浏览器打开  http://127.0.0.1:8000

依赖文件：
    conditional_transformer_best.pt   （与本文件放在同一目录）
"""

import io
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhconv import convert
from xpinyin import Pinyin

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from music21 import stream, note, tempo, metadata, meter, spanner

# ----------------------------------------------------------------------------
# 0. 基础常量
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "conditional_transformer_best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

PUNCT_COMMA = "，,、"
PUNCT_PERIOD = "。.！？!?；;"

pyin = Pinyin()

# 特殊 token id（与词表顺序一致：<PAD>=0, <BOS>=1, <EOS>=2）
PAD = 0
BOS = 1
EOS = 2


def get_tone(c):
    """取单字声调（1-4），取不到记 0。与训练时一致。"""
    p = pyin.get_pinyin(convert(c, "zh-cn"), tone_marks="numbers")
    import re

    m = re.search(r"[1-4]", p)
    return int(m.group()) if m else 0


# ----------------------------------------------------------------------------
# 1. 模型结构（与 notebook「四、模型」完全一致）
# ----------------------------------------------------------------------------
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.pitch_head = nn.Linear(d_model, num_pitches)
        self.dur_head = nn.Linear(d_model, num_durs)
        self.char_end_head = nn.Linear(d_model, 1)

    def forward(self, pitch_in, dur_in, tone_cond, word_start_cond, gong_id):
        B, T = pitch_in.shape

        pos = torch.arange(T, device=pitch_in.device).unsqueeze(0).expand(B, T)
        gong = self.gong_emb(gong_id).unsqueeze(1).expand(B, T, -1)

        h = (
            self.pitch_emb(pitch_in)
            + self.dur_emb(dur_in)
            + self.tone_emb(tone_cond)
            + self.word_start_emb(word_start_cond)
            + gong
            + self.pos_emb(pos)
        )

        causal_mask = torch.triu(
            torch.ones(T, T, device=pitch_in.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = pitch_in.eq(PAD)

        z = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)

        pitch_logits = self.pitch_head(z)
        dur_logits = self.dur_head(z)
        char_end_logits = self.char_end_head(z).squeeze(-1)

        return pitch_logits, dur_logits, char_end_logits


# ----------------------------------------------------------------------------
# 2. 加载组员训练好的模型（d_model=192, num_layers=4）
# ----------------------------------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

pitch_vocab = ckpt["pitch_vocab"]
dur_vocab = ckpt["dur_vocab"]
gongs = ckpt["gongs"]
gong2id = ckpt["gong2id"]
pitch2id = ckpt["pitch2id"]
dur2id = ckpt["dur2id"]

model = ConditionalMelodyTransformer(
    num_pitches=len(pitch_vocab),
    num_durs=len(dur_vocab),
    num_gongs=len(gongs),
    d_model=192,
    nhead=8,
    num_layers=4,
    dropout=0.15,
).to(device)

model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print(f"[加载完成] 模型参数量 {sum(p.numel() for p in model.parameters()):,} | 宫调 {len(gongs)} 个")


# ----------------------------------------------------------------------------
# 3. 生成逻辑（复用 notebook「六、生成」）
# ----------------------------------------------------------------------------
def top_p_sample(logits, top_p=0.9, temperature=0.85, banned_ids=None):
    logits = logits / max(temperature, 0.1)

    if banned_ids is not None:
        logits = logits.clone()
        for bid in banned_ids:
            logits[bid] = -1e9

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    keep = cum_probs <= top_p
    keep[0] = True

    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    filtered = filtered / filtered.sum().clamp(min=1e-8)

    return torch.multinomial(filtered, 1).item()


def generate_conditional(
    gongdiao,
    lyrics,
    seed=42,
    temperature=0.85,
    top_p=0.9,
    max_notes_per_char=4,
    end_threshold=0.5,
):
    """与 notebook「六、生成」逐字一致。
    关键点：条件序列(tone/word_start)比 pitch/dur 序列长 1，
    模型借此错位"看到"当前待预测位置的条件，切勿额外补 PAD。"""
    torch.manual_seed(seed)
    import random

    random.seed(seed)
    model.eval()

    if gongdiao not in gong2id:
        gongdiao = gongs[0]

    gong_id = torch.tensor([gong2id[gongdiao]], dtype=torch.long, device=device)

    pitch_ids = [BOS]
    dur_ids = [BOS]
    hist_tones = []
    hist_word_starts = []

    result = []

    banned_pitch = [PAD, BOS]
    banned_dur = [PAD, BOS]

    for ch in lyrics:
        if ch in PUNCT_COMMA or ch in PUNCT_PERIOD:
            result.append((ch, None))
            continue
        if not ch.strip():
            continue

        tone = get_tone(ch)
        if tone not in [0, 1, 2, 3, 4]:
            tone = 0

        notes = []

        for k in range(max_notes_per_char):
            pitch_in = torch.tensor([pitch_ids], dtype=torch.long, device=device)
            dur_in = torch.tensor([dur_ids], dtype=torch.long, device=device)

            current_word_start = 1 if k == 0 else 0

            tone_cond = torch.tensor(
                [hist_tones + [tone]], dtype=torch.long, device=device
            )
            word_start_cond = torch.tensor(
                [hist_word_starts + [current_word_start]],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                pitch_logits, dur_logits, end_logits = model(
                    pitch_in, dur_in, tone_cond, word_start_cond, gong_id
                )

            next_pitch_id = top_p_sample(
                pitch_logits[0, -1],
                top_p=top_p,
                temperature=temperature,
                banned_ids=banned_pitch,
            )
            next_dur_id = top_p_sample(
                dur_logits[0, -1],
                top_p=top_p,
                temperature=temperature,
                banned_ids=banned_dur,
            )

            if next_pitch_id == EOS or next_dur_id == EOS:
                if len(notes) == 0:
                    continue
                break

            midi = pitch_vocab[next_pitch_id]
            dur = dur_vocab[next_dur_id]

            notes.append((int(midi), float(dur)))

            pitch_ids.append(next_pitch_id)
            dur_ids.append(next_dur_id)
            hist_tones.append(tone)
            hist_word_starts.append(current_word_start)

            end_prob = torch.sigmoid(end_logits[0, -1]).item()

            # 至少生成一个音后，再允许模型判断这个字结束
            if k >= 1 and end_prob > end_threshold:
                break

        if len(notes) == 0:
            notes = [(60, 1.0)]  # 极少数兜底

        result.append((ch, notes))

    return result


# ----------------------------------------------------------------------------
# 4. 导出 MusicXML（复用 notebook「七、导出」）
# ----------------------------------------------------------------------------
def build_musicxml_bytes(mel, title="AI生成"):
    s = stream.Stream()
    s.insert(0, metadata.Metadata())
    s.metadata.title = title

    s.append(tempo.MetronomeMark(number=72))
    s.append(meter.TimeSignature("4/4"))

    for idx, (ch, notes) in enumerate(mel):
        if notes is None:
            if ch in PUNCT_PERIOD:
                s.append(note.Rest(quarterLength=1.0))
            else:
                s.append(note.Rest(quarterLength=0.5))
            continue

        end = idx + 1 < len(mel) and mel[idx + 1][1] is None
        ptype = "period" if end and mel[idx + 1][0] in PUNCT_PERIOD else "comma"

        made = []
        for j, (m, d) in enumerate(notes):
            nt = note.Note(m)
            nt.quarterLength = d
            if end and j == len(notes) - 1:
                nt.quarterLength = 3.0 if ptype == "period" else 2.0
            if j == 0:
                nt.lyric = ch
            made.append(nt)
            s.append(nt)

        if len(made) > 1:
            s.insert(0, spanner.Slur(made))

    s = s.makeMeasures()

    # music21 写文件后读回字节（最稳妥，兼容所有版本）
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
        tmp = f.name
    s.write("musicxml", fp=tmp)
    data = open(tmp, "rb").read()
    os.remove(tmp)
    return data


# ----------------------------------------------------------------------------
# 5. Web 服务
# ----------------------------------------------------------------------------
app = FastAPI(title="九宫旋律生成")


class GenReq(BaseModel):
    lyrics: str
    gongdiao: str
    title: str = "AI生成旋律"
    seed: int = 42
    temperature: float = 0.85


@app.get("/api/gongs")
def list_gongs():
    return {"gongs": gongs}


@app.post("/api/generate")
def generate(req: GenReq):
    import base64
    import music21

    lyrics = (req.lyrics or "").strip()
    if not lyrics:
        return JSONResponse({"error": "歌词不能为空"}, status_code=400)

    mel = generate_conditional(
        req.gongdiao,
        lyrics,
        seed=req.seed,
        temperature=req.temperature,
    )

    n_notes = sum(len(notes) for _, notes in mel if notes)
    if n_notes == 0:
        return JSONResponse({"error": "未能生成有效音符，请换一段歌词试试"}, status_code=400)

    # 整理成前端易于展示的结构：每个「字」一组音符（或一个停顿）
    items = []
    for ch, notes in mel:
        if notes is None:
            items.append({"char": ch, "rest": True})
        else:
            items.append({
                "char": ch,
                "rest": False,
                "notes": [
                    {
                        "midi": int(m),
                        "name": music21.pitch.Pitch(int(m)).nameWithOctave,
                        "dur": float(d),
                    }
                    for m, d in notes
                ],
            })

    xml = build_musicxml_bytes(mel, title=req.title)
    xml_b64 = base64.b64encode(xml).decode("ascii")

    return JSONResponse({
        "title": req.title,
        "gongdiao": req.gongdiao,
        "note_count": n_notes,
        "items": items,
        "musicxml_base64": xml_b64,
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
