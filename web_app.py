#!/usr/bin/env python
"""九宫大成诗词音乐生成 — Web 界面"""

import sys, os, json, re, time, base64, math
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.config import DeepSeekConfig
from src.deepseek_client import DeepSeekClient
from src.rich_prompt import RichPromptBuilder
from src.prompt_templates import BARE_SYSTEM_PROMPT, build_bare_task
from src.gongche_vocab import parse_compact_gongche
from src.musicxml_writer import gongche_to_musicxml
from src.data_loader import get_dataset as _get_ds
from src.quantitative_eval import QuantitativeEvaluator, eval_result_to_dict
from src.transformer_generator import load_generator as _load_tgen, list_gongs as _list_gongs

# ---- 全局初始化 ----
print("Loading...")
with open(Path(__file__).parent / "features_cache.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)
_ds = _get_ds()

# Build sorted qupai list: by song_count desc, with count label
QUPAI_DATA = []
for name, feat in FEATURES["qupai_features"].items():
    cnt = feat.get("song_count", 1)
    region = list(feat.get("region_distribution", {}).keys())
    region_tag = region[0][:1] if region else ""
    mode = list(feat.get("mode_distribution", {}).keys())
    mode_str = mode[0] if mode else ""
    QUPAI_DATA.append({
        "name": name,
        "count": cnt,
        "region": region_tag,
        "mode": mode_str,
    })
QUPAI_DATA.sort(key=lambda x: -x["count"])
QUPAI_JSON = json.dumps(QUPAI_DATA, ensure_ascii=False)
_gongs_list = _list_gongs()   # 仅读 checkpoint 元数据，不加载模型
GONGS_JSON = json.dumps(_gongs_list, ensure_ascii=False)
print(f"Ready: {len(QUPAI_DATA)} qupai, {len(_gongs_list)} gongdiao")

# 评估器（延迟初始化）
_evaluator = None

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = QuantitativeEvaluator(FEATURES)
    return _evaluator

# Transformer 生成器（延迟加载）
_tgen = None
def get_transformer():
    global _tgen
    if _tgen is None:
        _tgen = _load_tgen()
    return _tgen

# 宫调列表（启动时已加载）
def get_gongs():
    return _gongs_list
    return _GONGS


def _compute_line_indices(lyrics: str) -> list:
    """计算歌词行边界索引（每个行末字在整段去标点文本中的位置）。

    返回: [行1末字索引, 行2末字索引, ...] 不包含最后一行的末字。
    """
    clean = re.sub(r'[，,。！!？?；;、\s\n\r\t]', '', lyrics)
    if not clean:
        return []
    # 按标点/换行拆分原始歌词，计算每行的字符位置
    lines_raw = re.split(r'[，,。！!？?；;、\n\r]+', lyrics.strip())
    lines_raw = [ln.strip() for ln in lines_raw if ln.strip()]
    if len(lines_raw) <= 1:
        return []
    indices = []
    pos = 0
    for i, line in enumerate(lines_raw):
        # 计算该行在 clean 字符串中的字符数
        line_clean = re.sub(r'[，,。！!？?；;、\s\n\r\t]', '', line)
        pos += len(line_clean)
        if i < len(lines_raw) - 1:
            indices.append(pos - 1)  # 行末字的索引
    return indices


def get_examples(qupai, n=2):
    songs = _ds.get_songs_by_qupai(qupai)
    return [{"polyu_id": s.polyu_id, "lyrics": s.lyrics_text,
             "lyric_note_groups": s.lyric_note_groups,
             "note_count": s.note_count} for s in songs[:n]]


def generate(qupai, lyrics):
    builder = RichPromptBuilder(FEATURES)
    examples = get_examples(qupai, 2)
    system_prompt, user_prompt = builder.build(qupai=qupai, lyrics=lyrics, examples=examples)

    client_config = DeepSeekConfig(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    client = DeepSeekClient(client_config)
    t0 = time.time()
    raw, usage = client.chat(system_prompt, user_prompt, max_tokens=800)
    api_time = time.time() - t0

    gc_groups = parse_compact_gongche(raw)

    mxl_b64 = ""
    saved_xml_path = ""
    if gc_groups:
        out_dir = Path("experiments/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_qp = re.sub(r'[《》\s]', '', qupai)
        safe_ly = re.sub(r'[《》\s\n，,。！!？?；;、]', '', lyrics)[:10]
        saved_xml_path = str(out_dir / f"{safe_qp}_{safe_ly}.musicxml")
        # 计算句间休止位置
        line_indices = _compute_line_indices(lyrics)
        try:
            gongche_to_musicxml(lyric_groups=gc_groups, output_path=saved_xml_path,
                                title=f"AI: {qupai}", work_title=f"JiuGong: {qupai}", qupai=qupai,
                                line_indices=line_indices)
            with open(saved_xml_path, "rb") as f:
                mxl_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            saved_xml_path = ""

    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "raw_response": raw,
            "musicxml_b64": mxl_b64, "saved_xml_path": saved_xml_path,
            "tokens": usage.get("total_tokens", 0),
            "api_time": round(api_time, 2), "n_groups": len(gc_groups),
            "gc_groups": gc_groups}

def generate_bare(qupai, lyrics):
    """极简提示词生成（无曲牌特征/声调规则/样例/风格指导）"""
    system_prompt = BARE_SYSTEM_PROMPT
    user_prompt = build_bare_task(qupai, lyrics)

    client_config = DeepSeekConfig(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    client = DeepSeekClient(client_config)
    t0 = time.time()
    raw, usage = client.chat(system_prompt, user_prompt, max_tokens=800)
    api_time = time.time() - t0

    gc_groups = parse_compact_gongche(raw)

    mxl_b64 = ""
    saved_xml_path = ""
    if gc_groups:
        out_dir = Path("experiments/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_qp = re.sub(r'[《》\s]', '', qupai)
        safe_ly = re.sub(r'[《》\s\n，,。！!？?；;、]', '', lyrics)[:10]
        saved_xml_path = str(out_dir / f"bare_{safe_qp}_{safe_ly}.musicxml")
        line_indices = _compute_line_indices(lyrics)
        try:
            gongche_to_musicxml(lyric_groups=gc_groups, output_path=saved_xml_path,
                                title=f"Bare: {qupai}", work_title=f"JiuGong-Bare: {qupai}", qupai=qupai,
                                line_indices=line_indices)
            with open(saved_xml_path, "rb") as f:
                mxl_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            saved_xml_path = ""

    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "raw_response": raw,
            "musicxml_b64": mxl_b64, "saved_xml_path": saved_xml_path,
            "tokens": usage.get("total_tokens", 0),
            "api_time": round(api_time, 2), "n_groups": len(gc_groups),
            "gc_groups": gc_groups}


def generate_transformer(lyrics, gongdiao, title="AI生成旋律"):
    """条件式 Transformer 直接生成旋律"""
    t0 = time.time()
    gen = get_transformer()
    gc_groups = gen.generate(lyrics, gongdiao)
    api_time = time.time() - t0
    n_notes = sum(len(g.notes) for g in gc_groups)

    # 图形乐谱 items
    items = gen.groups_to_items(gc_groups)

    mxl_b64 = ""
    saved_xml_path = ""
    if gc_groups:
        out_dir = Path("experiments/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_go = re.sub(r'[《》\s]', '', gongdiao)
        safe_ly = re.sub(r'[《》\s\n，,。！!？?；;、]', '', lyrics)[:10]
        saved_xml_path = str(out_dir / f"trans_{safe_go}_{safe_ly}.musicxml")
        line_indices = _compute_line_indices(lyrics)
        try:
            gongche_to_musicxml(lyric_groups=gc_groups, output_path=saved_xml_path,
                                title=title, work_title="JiuGong-Transformer",
                                qupai=gongdiao, line_indices=line_indices)
            with open(saved_xml_path, "rb") as f:
                mxl_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            saved_xml_path = ""

    return {
        "raw_response": "\n".join(
            f"{g.lyric}[{' '.join(n.gongche for n in g.notes)}]"
            for g in gc_groups
        ),
        "musicxml_b64": mxl_b64, "saved_xml_path": saved_xml_path,
        "tokens": n_notes, "api_time": round(api_time, 2),
        "n_groups": len(gc_groups), "gc_groups": gc_groups,
        "items": items, "title": title,
    }


def evaluate_generated(gc_groups, lyrics, qupai):
    """对生成结果进行定量评估（非致命：失败不影响生成）"""
    try:
        ev = get_evaluator()
        result = ev.evaluate(
            generated_groups=gc_groups,
            lyrics=lyrics,
            qupai=qupai,
        )
        out_dir = Path("experiments/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_qp = re.sub(r'[《》\s]', '', qupai)
        safe_ly = re.sub(r'[《》\s\n，,。！!？?；;、]', '', lyrics)[:10]
        eval_path = out_dir / f"{safe_qp}_{safe_ly}.eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result_to_dict(result), f, ensure_ascii=False, indent=2)
        return eval_result_to_dict(result)
    except (ValueError, KeyError, TypeError, OSError, ImportError):
        import traceback
        traceback.print_exc()
        return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>九宫大成 · 诗词音乐生成</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","Noto Sans SC",sans-serif;background:#f5f0e8;color:#3e2c1c;min-height:100vh}
.header{background:linear-gradient(135deg,#8b4513,#a0522d 50%,#6b3410);color:#f5e6d0;padding:20px 32px;text-align:center;border-bottom:4px solid #d4a574}
.header h1{font-size:26px;letter-spacing:4px}
.container{max-width:1400px;margin:0 auto;padding:16px;display:grid;grid-template-columns:400px 1fr;gap:16px}
@media(max-width:960px){.container{grid-template-columns:1fr}}
.card{background:#fffaf3;border:1px solid #d4c4a8;border-radius:10px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card h2{font-size:15px;color:#6b3410;border-bottom:2px solid #d4a574;padding-bottom:6px;margin-bottom:12px}
label{display:block;font-size:12px;color:#5c3a1e;margin-bottom:3px;margin-top:10px;font-weight:bold}
input,select,textarea{width:100%;padding:9px 11px;border:1px solid #c4b494;border-radius:6px;font-size:14px;font-family:Microsoft YaHei,sans-serif;background:#fffef9}
select{height:auto;overflow-y:auto;cursor:pointer}
select option{padding:4px 6px;font-size:13px}
textarea{resize:vertical;min-height:80px;line-height:1.8}
.preview-lines{margin-top:8px;padding:10px 12px;background:#fef9f0;border-radius:6px;border:1px dashed #d4c4a8;font-size:13px;color:#5c2e0e;line-height:2;min-height:20px}
.preview-lines .ln{display:inline-block;background:#ede3d3;padding:1px 8px;margin:2px 3px;border-radius:3px}
.preview-lines .sep{color:#a08060;font-size:11px;margin:0 2px}
.btn{width:100%;padding:11px;margin-top:14px;background:#8b4513;color:#fff;border:none;border-radius:8px;font-size:15px;cursor:pointer;letter-spacing:2px;transition:background .2s}
.btn:hover{background:#a0522d}
.btn:disabled{background:#c4a890;cursor:not-allowed}
.tab-bar{display:flex;gap:3px;margin-bottom:10px;flex-wrap:wrap}
.tab{padding:7px 14px;border:1px solid #d4c4a8;border-radius:6px 6px 0 0;background:#ede3d3;cursor:pointer;font-size:12px;user-select:none}
.tab.active{background:#fffaf3;border-bottom-color:#fffaf3;font-weight:bold;color:#6b3410}
.tab-content{display:none;padding:8px 0}
.tab-content.active{display:block}
pre{background:#2d2418;color:#e8dcc8;padding:14px;border-radius:8px;overflow-x:auto;font-size:12px;line-height:1.6;max-height:460px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}
.gongche-display{background:#fdf6ec;border:2px solid #d4a574;border-radius:10px;padding:18px;font-size:16px;line-height:2;letter-spacing:2px;color:#5c2e0e;max-height:500px;overflow-y:auto}
.status-bar{display:flex;gap:10px;font-size:12px;color:#8b6914;flex-wrap:wrap}
.status-bar span{background:#fef3e0;padding:3px 9px;border-radius:4px}
.download-btn{display:inline-block;padding:7px 18px;background:#2d6b30;color:#fff;border-radius:6px;text-decoration:none;font-size:13px;margin:4px 6px 0 0}
.download-btn:hover{background:#3d8b40}
.download-btn.alt{background:#6b3410}
.lyric-line{margin:2px 0;padding:3px 8px;border-left:3px solid #d4a574;background:#fef9f0;font-weight:bold}
.melody-line{padding-left:16px;margin-bottom:5px;font-size:15px;color:#5c2e0e}
#status-line{margin-top:6px;font-size:12px;color:#8b6914}
.loading{display:inline-block;animation:spin 1s linear infinite}@keyframes spin{100%{transform:rotate(360deg)}}
.qupai-dropdown{position:absolute;top:100%;left:0;right:0;max-height:260px;overflow-y:auto;background:#fffaf3;border:1px solid #c4b494;border-top:none;border-radius:0 0 8px 8px;z-index:100;box-shadow:0 4px 12px rgba(0,0,0,.12)}
.qupai-dropdown-item{padding:6px 10px;cursor:pointer;font-size:13px;border-bottom:1px solid #eee;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.qupai-dropdown-item:hover,.qupai-dropdown-item.active{background:#fef3e0;color:#6b3410}
.qupai-dropdown-item .cnt{font-size:10px;color:#a08060;margin-left:6px}
/* Transformer 图形乐谱 */
.t-score{display:flex;align-items:flex-end;gap:2px;min-height:200px;overflow-x:auto;padding:12px 0}
.t-col{display:flex;flex-direction:column-reverse;align-items:center;min-width:36px}
.t-col .t-note{width:32px;text-align:center;font-size:10px;border-radius:3px;margin:1px 0;padding:2px 0;
  color:#fff;font-weight:600;line-height:1.3}
.t-col .t-char{border-top:2px solid #8a5a2b;padding-top:4px;font-size:13px;font-weight:bold;color:#5c3a1e}
.t-col.rest .t-note{background:transparent;color:#ccc;font-size:16px}
.t-col.rest .t-char{color:#cbb89e;border-top-color:#ddd}
</style>
</head>
<body>

<div class="header">
<h1>🏮 九宫大成 · 诗词音乐生成</h1>
<p>选择曲牌 + 输入歌词 → AI 生成工尺谱旋律 + 五线谱 MusicXML</p>
</div>

<div class="container">
<div>
<div class="card">
<h2>📝 创作参数</h2>
<label>生成引擎</label>
<div style="display:flex;gap:16px;margin-bottom:10px;font-size:13px">
<label style="display:inline;font-weight:normal;cursor:pointer"><input type="radio" name="engine" value="llm" checked onchange="switchEngine()"> 🤖 LLM + Prompt（DeepSeek API，选曲牌）</label>
<label style="display:inline;font-weight:normal;cursor:pointer"><input type="radio" name="engine" value="transformer" onchange="switchEngine()"> 🧠 Transformer 模型（本地推理，选宫调）</label>
</div>
<div id="llm-controls">
<label>曲牌名 <span style="font-weight:normal;color:#a08060">（输入1-2字搜索，点击下拉选取）</span></label>
<div style="position:relative">
<input type="text" id="qupai-filter" placeholder="输入曲牌名搜索..." autocomplete="off" oninput="filterQupai()" onfocus="filterQupai()" onkeydown="handleQupaiKey(event)">
<div id="qupai-dropdown" class="qupai-dropdown" style="display:none"></div>
</div>
<input type="hidden" id="qupai" value="">
<div style="font-size:12px;color:#6b3410;margin-top:4px;padding:6px 10px;background:#fef9f0;border-radius:6px;border:1px solid #e0d5c0;min-height:20px" id="qupai-info">
<span style="color:#a08060">尚未选择曲牌</span>
</div>
</div>
<div id="transformer-controls" style="display:none">
<div style="display:flex;gap:12px">
<div style="flex:1"><label>宫调</label>
<select id="gongdiao" size="5" style="width:100%"></select>
<div style="font-size:11px;color:#8b6914;margin-top:2px" id="gongdiao-info"></div></div>
<div style="flex:1"><label>曲名</label>
<input type="text" id="trans-title" value="AI生成旋律" style="width:100%" /></div>
</div>
</div>

<label>歌词内容（自动按标点换行）</label>
<textarea id="lyrics" placeholder="小院春寒，梨花半落胭脂雨。绿窗朱户，燕绕秋千柱。" oninput="previewLines()">小院春寒，梨花半落胭脂雨。绿窗朱户，燕绕秋千柱。心事轻梳，欲语还羞住。风起处，落红无数，谁共斜阳暮？</textarea>
<div class="preview-lines" id="line-preview"></div>

<div style="margin-top:8px;font-size:12px;display:flex;align-items:center;gap:6px" id="compare-row">
<input type="checkbox" id="compare-mode" style="width:auto;cursor:pointer">
<label for="compare-mode" style="display:inline;margin:0;cursor:pointer;font-weight:normal">
对比模式（Bare无提示词 + Rich提示词工程，双路生成后并排对比）
</label>
</div>
<button class="btn" id="gen-btn" onclick="doGenerate()">🎵 生成音乐</button>
<div id="status-line"></div>
</div>

<div class="card" style="margin-top:12px">
<h2>📊 关于</h2>
<div style="font-size:12px;line-height:1.8;color:#6b4e2e">
《九宫大成南北词宫谱》<br>6,563首 · 2,053曲牌 · 73宫调<br>南词2,744 · 北词3,428<br>
<span style="color:#8b6914">DeepSeek API 驱动</span>
</div>
</div>
</div>

<div>
<div id="result-area" style="display:none">
<div class="card" style="margin-bottom:10px"><div class="status-bar" id="result-status"></div></div>
<div class="tab-bar">
<div class="tab active" onclick="switchTab('gongche')">🎼 工尺谱</div>
<div class="tab" onclick="switchTab('evaluation')">📊 定量评估</div>
<div class="tab" onclick="switchTab('prompt')">📋 提示词</div>
<div class="tab" onclick="switchTab('raw')">📝 模型响应</div>
<div class="tab" onclick="switchTab('features')">📊 曲牌特征</div>
</div>
<div id="tab-gongche" class="tab-content active card">
<h2>生成旋律 — 工尺谱</h2>
<div class="gongche-display" id="gongche-output"></div>
<div style="margin-top:10px">
<a class="download-btn" id="dl-mxl" href="#" download="generated.musicxml">📥 下载 MusicXML</a>
<a class="download-btn alt" id="dl-txt" href="#" download="generated.txt">📥 下载工尺谱</a>
</div>
<div id="trans-score-area" style="display:none">
<div class="t-score" id="t-score"></div>
<div style="font-size:11px;color:#8b6914;margin-top:4px" id="t-score-info"></div>
</div>
</div>
<div id="tab-evaluation" class="tab-content card">
<h2>定量评估报告</h2>
<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">
<div style="flex:1;min-width:110px;text-align:center;padding:14px;background:#fef3e0;border-radius:8px;border:2px solid #d4a574">
<div style="font-size:28px;font-weight:bold;color:#8b4513" id="eval-overall">--</div>
<div style="font-size:11px;color:#8b6914">综合评分</div>
</div>
<div style="flex:1;min-width:110px;text-align:center;padding:14px;background:#e8f5e9;border-radius:8px;border:2px solid #66bb6a">
<div style="font-size:28px;font-weight:bold;color:#2e7d32" id="eval-style">--</div>
<div style="font-size:11px;color:#558b2f">风格相似度</div>
</div>
<div style="flex:1;min-width:110px;text-align:center;padding:14px;background:#e3f2fd;border-radius:8px;border:2px solid #42a5f5">
<div style="font-size:28px;font-weight:bold;color:#1565c0" id="eval-tone">--</div>
<div style="font-size:11px;color:#1976d2">声调综合</div>
<div style="font-size:9px;color:#64b5f6;margin-top:3px">
  规则 <span id="eval-tone-rule">--</span> / 数据 <span id="eval-tone-data">--</span>
</div>
</div>
</div>
<h3 style="margin-top:12px;font-size:14px">风格相似度 — 分项指标</h3>
<div id="eval-style-detail" style="font-size:12px;line-height:2"></div>
<h3 style="margin-top:12px;font-size:14px">声调对齐度 — 综合评分（规则50%+数据50%）</h3>
<div id="eval-tone-detail" style="font-size:12px;line-height:2"></div>
<h3 style="margin-top:12px;font-size:13px">声调对齐度 — 分方法对比</h3>
<div style="display:flex;gap:16px;flex-wrap:wrap">
<div style="flex:1;min-width:200px;padding:8px;background:#fff8e1;border-radius:6px" id="eval-tone-rule-detail"></div>
<div style="flex:1;min-width:200px;padding:8px;background:#e8f0fe;border-radius:6px" id="eval-tone-data-detail"></div>
</div>
<details style="margin-top:12px">
<summary style="cursor:pointer;font-weight:bold;color:#6b3410">逐字声调分析 (点击展开)</summary>
<div id="eval-char-detail" style="max-height:300px;overflow-y:auto;font-size:11px;margin-top:6px"></div>
</details>
<p style="font-size:10px;color:#a08060;margin-top:10px" id="eval-meta"></p>
<details style="margin-top:12px">
<summary style="cursor:pointer;font-weight:bold;color:#6b3410">📖 指标计算说明 (点击展开)</summary>
<div id="eval-method-detail" style="font-size:11px;line-height:1.8;max-height:400px;overflow-y:auto;margin-top:6px;padding:8px;background:#fefef8;border-radius:6px;border:1px solid #e0d5c0"></div>
</details>
</div>
<div id="tab-prompt" class="tab-content card">
<h2>System Prompt</h2><pre id="sys-prompt"></pre>
<h2 style="margin-top:14px">User Prompt</h2><pre id="usr-prompt"></pre>
</div>
<div id="tab-raw" class="tab-content card">
<h2>模型原始输出</h2><pre id="raw-resp"></pre>
</div>
<div id="tab-features" class="tab-content card">
<h2>曲牌特征数据</h2><pre id="feat-data"></pre>
</div>
</div>
</div>
</div>

<div id="compare-result-area" style="display:none">
<div class="card" style="border:2px solid #ff9800;margin-bottom:12px">
<h2>Bare vs Rich — 提示词工程提升对比</h2>
<div style="font-size:11px;color:#888;margin-bottom:8px" id="compare-status"></div>
<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
<div style="flex:1;min-width:80px;text-align:center;padding:8px;background:#fff8e1;border-radius:6px">
<div style="font-size:20px;font-weight:bold;color:#e65100" id="bare-overall">--</div><div style="font-size:10px">Bare综合</div></div>
<div style="flex:1;min-width:80px;text-align:center;padding:8px;background:#fef3e0;border-radius:6px">
<div style="font-size:20px;font-weight:bold;color:#8b4513" id="rich-overall">--</div><div style="font-size:10px">Rich综合</div></div>
<div style="flex:1;min-width:80px;text-align:center;padding:8px;background:#e8f5e9;border-radius:6px">
<div style="font-size:20px;font-weight:bold;color:#2e7d32" id="delta-overall">--</div><div style="font-size:10px">提升</div></div>
</div>
<div style="font-size:10px;color:#888;margin-bottom:4px" id="bare-meta"></div>
<div class="gongche-display" style="max-height:160px;font-size:13px;margin-bottom:10px" id="bare-gongche"></div>
<a class="download-btn alt" id="dl-bare-mxl" href="#" download="bare.musicxml">Download MusicXML (Bare)</a>
<div id="compare-delta-table" style="margin-top:12px"></div>
</div>
</div>

<script>
var QDATA = __QUPAI_DATA__;
var GONGS = __GONGS_DATA__;
var sel = document.getElementById("qupai");
var filterInput = document.getElementById("qupai-filter");
var infoDiv = document.getElementById("qupai-info");
var dropdown = document.getElementById("qupai-dropdown");
var dropdownIndex = -1;

// 当前已选曲牌
var selectedQupai = null;

function selectQupai(name) {
  // 从 QDATA 查找完整信息
  var found = QDATA.filter(function(q){ return q.name === name; })[0];
  if (found) {
    selectedQupai = found;
    sel.value = found.name;
    updateInfo();
  }
  hideDropdown();
  filterInput.value = name || '';
  filterInput.focus();
}

function hideDropdown() {
  dropdown.style.display = "none";
  dropdown.innerHTML = "";
  dropdownIndex = -1;
}

function renderQupaiDropdown(items) {
  if (items.length === 0) {
    hideDropdown();
    return;
  }
  var show = items.slice(0, 30);
  var html = "";
  show.forEach(function(q, i){
    html += '<div class="qupai-dropdown-item' + (i===0?' active':'') + '" data-name="' + escHtml(q.name) + '">' +
      escHtml(q.name) + '<span class="cnt">' + q.count + '首 ' + q.region + (q.mode?' '+q.mode:'') + '</span></div>';
  });
  dropdown.innerHTML = html;
  dropdown.style.display = "block";
  dropdownIndex = -1;
}

function filterQupai() {
  var val = filterInput.value.trim().toLowerCase();
  if (!val) {
    // 未输入时展示热门曲牌，方便直接点选
    renderQupaiDropdown(QDATA);
    return;
  }
  // 筛选匹配项
  var filtered = QDATA.filter(function(q){
    return q.name.toLowerCase().indexOf(val) >= 0;
  });
  renderQupaiDropdown(filtered);
}

// 下拉项点击（事件委托）
dropdown.addEventListener('mousedown', function(e){
  var item = e.target.closest('.qupai-dropdown-item');
  if (item) {
    e.preventDefault();
    e.stopPropagation();
    var name = item.getAttribute('data-name');
    if (name) selectQupai(name);
  }
});

// 键盘导航
function handleQupaiKey(e) {
  var items = dropdown.querySelectorAll('.qupai-dropdown-item');
  if (items.length === 0) return;
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    dropdownIndex = Math.min(dropdownIndex + 1, items.length - 1);
    updateDropdownActive(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    dropdownIndex = Math.max(dropdownIndex - 1, 0);
    updateDropdownActive(items);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (dropdownIndex >= 0 && dropdownIndex < items.length) {
      var name = items[dropdownIndex].getAttribute('data-name');
      if (name) selectQupai(name);
    } else if (items.length > 0) {
      var n = items[0].getAttribute('data-name');
      if (n) selectQupai(n);
    }
  } else if (e.key === 'Escape') {
    hideDropdown();
  }
}

function updateDropdownActive(items) {
  items.forEach(function(item, i){
    if (i === dropdownIndex) {
      item.classList.add('active');
      item.scrollIntoView({block:'nearest'});
    } else {
      item.classList.remove('active');
    }
  });
}

function updateInfo() {
  if (selectedQupai) {
    infoDiv.innerHTML = '<b>' + escHtml(selectedQupai.name) + '</b> ' +
      '<span style="font-size:10px;color:#8b6914">' +
      '数据集收录 <b>' + selectedQupai.count + '</b> 首 | ' +
      '板块: ' + escHtml(selectedQupai.region||'?') +
      (selectedQupai.mode ? ' | 宫调: ' + escHtml(selectedQupai.mode) : '') +
      '</span>';
  } else {
    infoDiv.innerHTML = '<span style="color:#a08060">尚未选择曲牌</span>';
  }
}

// 点击页面其他地方关闭下拉
document.addEventListener('click', function(e){
  if (e.target !== filterInput && e.target !== dropdown && !dropdown.contains(e.target)) {
    hideDropdown();
  }
});

// 默认选择数据集中曲目数量最多的曲牌
selectQupai(QDATA[0].name);
updateInfo();

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// Gongdiao select
var gongdiaoSel = document.getElementById("gongdiao");
GONGS.forEach(function(g, i){
  var o = document.createElement("option");
  o.value = g; o.textContent = g;
  if (i === 0) o.selected = true;
  gongdiaoSel.appendChild(o);
});
gongdiaoSel.addEventListener("change", function(){
  document.getElementById("gongdiao-info").textContent = "已选: " + gongdiaoSel.value;
});
document.getElementById("gongdiao-info").textContent = "已选: " + GONGS[0];

// Engine switching
function switchEngine() {
  var isTrans = document.querySelector('input[name="engine"]:checked').value === "transformer";
  document.getElementById("llm-controls").style.display = isTrans ? "none" : "block";
  document.getElementById("transformer-controls").style.display = isTrans ? "block" : "none";
  document.getElementById("compare-row").style.display = isTrans ? "none" : "block";
}
switchEngine();

function splitLyrics(text) {
  var withBreaks = text.replace(/[，,。！!？?；;、]/g, '\n');
  return withBreaks.split(/\n+/).filter(function(s){return s.trim().length>0});
}
function previewLines() {
  var text = document.getElementById("lyrics").value;
  var lines = splitLyrics(text);
  var preview = document.getElementById("line-preview");
  if (lines.length > 1) {
    var html = lines.map(function(ln,i){return '<span class="ln">'+escHtml(ln)+'</span>'+(i<lines.length-1?'<span class="sep">→</span>':'');}).join('');
    preview.innerHTML = '将分为 <b>'+lines.length+'</b> 行: '+html;
  } else if (lines.length === 1) {
    preview.innerHTML = '单行歌词，共 '+lines[0].length+' 字';
  } else {
    preview.innerHTML = '';
  }
}
previewLines();

var lastResult = null;
async function doGenerate(){
var isTrans = document.querySelector('input[name="engine"]:checked').value === "transformer";
var rawText = document.getElementById("lyrics").value.trim();
var lines = splitLyrics(rawText);
var l = lines.join("\n");
if(!l){alert("请填写歌词");return}

var b = document.getElementById("gen-btn");
b.disabled = true;
b.textContent = isTrans ? "生成中(Transformer)..." : "生成中...";
document.getElementById("status-line").innerHTML = "<span class='loading'>⏳ 正在生成...</span>";

try{
var reqBody = {lyrics:l};
if (isTrans) {
  reqBody.mode = "transformer";
  reqBody.gongdiao = document.getElementById("gongdiao").value;
  reqBody.title = document.getElementById("trans-title").value.trim() || "AI生成旋律";
} else {
  var q = sel.value.trim();
  if (!q) { alert("请先选择一个曲牌"); b.disabled=false; b.textContent="🎵 生成音乐"; return; }
  reqBody.qupai = q;
  var isCompare = document.getElementById("compare-mode").checked;
  if (isCompare) reqBody.compare = true;
  b.textContent = isCompare ? "生成中(对比)..." : "生成中...";
}
var r = await fetch("/api/generate",{
  method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify(reqBody)
});

if(!r.ok){alert("HTTP "+r.status+": 服务异常，请稍后重试");b.disabled=false;b.textContent="🎵 生成音乐";return}

var d = await r.json();
if(d.error){alert(d.error);b.disabled=false;b.textContent="🎵 生成音乐";return}

lastResult = d;
if (d.mode === "transformer") {
  document.getElementById("compare-result-area").style.display = "none";
  document.getElementById("result-area").style.display = "block";
  document.getElementById("status-line").innerHTML = "✅ Transformer 生成完成 · "+d.tokens+" 音符 · "+d.api_time+"s · "+d.n_groups+" 字音组";
  document.getElementById("result-status").innerHTML =
    "<span>🧠 "+d.gongdiao+"</span><span>🎵 "+d.tokens+" 音符</span><span>⏱ "+d.api_time+"s</span><span>📐 "+d.n_groups+" 字音组</span>" +
    (d.saved_xml_path ? "<br><span style=\"font-size:11px\">📁 "+escHtml(d.saved_xml_path)+"</span>" : "");

  // 图形乐谱
  if (d.items && d.items.length > 0) {
    var ti = d.items;
    var allMidi = [];
    ti.forEach(function(it){ if(!it.rest) it.notes.forEach(function(n){ allMidi.push(n.midi); }); });
    var minM = allMidi.length ? Math.min.apply(null, allMidi) : 60;
    var maxM = allMidi.length ? Math.max.apply(null, allMidi) : 72;
    var span = Math.max(maxM - minM, 1);
    var noteBg = function(midi) {
      var t = (midi - minM) / span;
      var r, g, b;
      if (t < 0.33) { r=100; g=149; b=237; }  // 低音蓝
      else if (t < 0.66) { r=60; g=179; b=113; } // 中音绿
      else { r=205; g=92; b=92; } // 高音红
      return "rgb("+r+","+g+","+b+")";
    };
    var scoreHtml = '';
    ti.forEach(function(it){
      if (it.rest) {
        scoreHtml += '<div class="t-col rest"><div class="t-note">休</div><div class="t-char">'+escHtml(it.char)+'</div></div>';
      } else {
        var noteDivs = it.notes.map(function(n){
          return '<div class="t-note" style="background:'+noteBg(n.midi)+'" title="'+n.name+'">'+n.name.replace(/\d/,'')+'</div>';
        }).join('');
        scoreHtml += '<div class="t-col"><div class="t-notes">'+noteDivs+'</div><div class="t-char">'+escHtml(it.char)+'</div></div>';
      }
    });
    document.getElementById("t-score").innerHTML = scoreHtml;
    var firstNote = (ti[0] && ti[0].notes && ti[0].notes[0]) ? ti[0].notes[0].name : '?';
    var lastItem = (ti[ti.length-1] && ti[ti.length-1].notes) ? ti[ti.length-1].notes : [];
    var lastNote = lastItem.length ? lastItem[lastItem.length-1].name : '?';
    document.getElementById("t-score-info").textContent = "音域: "+firstNote+" ~ "+lastNote+" | 色调: 低音蓝→中音绿→高音红";
    document.getElementById("trans-score-area").style.display = "block";
  } else {
    document.getElementById("trans-score-area").style.display = "none";
  }

  // 工尺谱文本（折叠在下方）
  var raw = d.raw_response || "";
  document.getElementById("gongche-output").innerHTML =
    "<pre style=\"background:transparent;color:#5c2e0e;font-size:16px;line-height:2;letter-spacing:2px;max-height:500px;white-space:pre-wrap;padding:0;margin:0;\">" + escHtml(raw) + "</pre>";
  document.getElementById("sys-prompt").textContent = "（Transformer 模型直接生成，不使用 Prompt）";
  document.getElementById("usr-prompt").textContent = "宫调: " + d.gongdiao + " | 歌词: " + l;
  document.getElementById("raw-resp").textContent = d.raw_response;
  document.getElementById("feat-data").textContent = JSON.stringify({gongdiao: d.gongdiao, engine: "transformer"}, null, 2);
  if (d.musicxml_b64) {
    var mx = document.getElementById("dl-mxl");
    mx.href = "data:application/vnd.recordare.musicxml+xml;base64,"+d.musicxml_b64;
    mx.download = "trans_"+d.gongdiao+"_"+l.replace(/\n/g,"").slice(0,8)+".musicxml";
  }
  var tx = document.getElementById("dl-txt");
  tx.href = "data:text/plain;charset=utf-8,"+encodeURIComponent(d.raw_response);
  tx.download = "trans_"+d.gongdiao+"_"+l.replace(/\n/g,"").slice(0,8)+".txt";
  if (d.evaluation) { renderEval(d.evaluation, "Transformer | "); }
  switchTab("gongche");
  b.disabled = false;
  b.textContent = "🎵 生成音乐";
  return;
}
if (d.mode === "compare") {
  // === 上部：Rich 数据填充正常 Tab 区 ===
  document.getElementById("trans-score-area").style.display = "none";
  document.getElementById("result-area").style.display = "block";
  document.getElementById("compare-result-area").style.display = "block";

  // 状态栏
  document.getElementById("status-line").innerHTML = "Done | Bare:"+d.bare.tokens+"t "+d.bare.api_time+"s + Rich:"+d.rich.tokens+"t "+d.rich.api_time+"s";
  document.getElementById("result-status").innerHTML =
    "<span>Bare:"+d.bare.tokens+"t</span><span>Rich:"+d.rich.tokens+"t</span><span>Rich MusicXML</span>" +
    (d.rich.saved_xml_path ? "<br><span style=\"font-size:11px\">"+escHtml(d.rich.saved_xml_path)+"</span>" : "");

  // 工尺谱 Tab — Rich
  var rawRich = d.rich.raw_response || "";
  document.getElementById("gongche-output").innerHTML =
    "<pre style=\"background:transparent;color:#5c2e0e;font-size:16px;line-height:2;letter-spacing:2px;max-height:500px;white-space:pre-wrap;padding:0;margin:0;\">" + escHtml(rawRich) + "</pre>";

  // 提示词 / 原始响应 / 曲牌特征 Tab — Rich
  document.getElementById("sys-prompt").textContent = d.rich.system_prompt;
  document.getElementById("usr-prompt").textContent = d.rich.user_prompt;
  document.getElementById("raw-resp").textContent = d.rich.raw_response;
  document.getElementById("feat-data").textContent = JSON.stringify(d.qupai_features,null,2);

  // MusicXML 下载 — Rich
  if (d.rich.musicxml_b64) {
    var mx2 = document.getElementById("dl-mxl");
    mx2.href = "data:application/vnd.recordare.musicxml+xml;base64,"+d.rich.musicxml_b64;
    mx2.download = "rich_"+q+"_"+l.replace(/\n/g,"").slice(0,8)+".musicxml";
  }
  var tx2 = document.getElementById("dl-txt");
  tx2.href = "data:text/plain;charset=utf-8,"+encodeURIComponent(d.rich.raw_response);
  tx2.download = "rich_"+q+"_"+l.replace(/\n/g,"").slice(0,8)+".txt";

  // 定量评估 Tab — Rich（复用已有渲染函数）
  if (d.rich.evaluation) { renderEval(d.rich.evaluation, "Rich | "); }
  switchTab("gongche");

  // === 下部：Bare 对比卡片 ===
  var pctC = function(v){ return (v != null && v >= 0) ? (v*100).toFixed(0)+"%" : "N/A"; };
  var dStr = function(v){ if (v==null) return "N/A"; var s=v>=0?"+":""; return s+(v*100).toFixed(0)+"%"; };
  document.getElementById("compare-status").textContent = "Bare: "+d.bare.tokens+"t "+d.bare.api_time+"s | Rich: "+d.rich.tokens+"t "+d.rich.api_time+"s";

  var be = d.bare && d.bare.evaluation;
  var re = d.rich && d.rich.evaluation;
  if (be) {
    document.getElementById("bare-overall").textContent = pctC(be.overall_score);
    document.getElementById("rich-overall").textContent = pctC(re ? re.overall_score : 0);
    var deltaOv = d.comparison ? d.comparison.overall_delta : 0;
    document.getElementById("delta-overall").textContent = d.comparison ? dStr(deltaOv) : "N/A";
    document.getElementById("delta-overall").style.color = (deltaOv >= 0) ? "#2e7d32" : "#c62828";
    document.getElementById("bare-meta").textContent = "Bare工尺谱 (Token:"+d.bare.tokens+" | "+d.bare.api_time+"s | "+d.bare.n_groups+" groups)";
  }
  document.getElementById("bare-gongche").innerHTML = "<pre style=\"background:transparent;color:#5c2e0e;font-size:13px;line-height:2;letter-spacing:2px;max-height:160px;white-space:pre-wrap;padding:0;margin:0;\">" + escHtml(d.bare.raw_response||"") + "</pre>";
  if (d.bare.musicxml_b64) {
    var bm2 = document.getElementById("dl-bare-mxl");
    bm2.href = "data:application/vnd.recordare.musicxml+xml;base64,"+d.bare.musicxml_b64;
    bm2.download = "bare_"+q+"_"+l.replace(/\n/g,"").slice(0,8)+".musicxml";
  }

  // 差异表
  if (d.comparison) {
    var cmp = d.comparison;
    var sn = {"format_compliance":"格式合规","pitch_distribution":"音高分布","interval_distribution":"音程分布","density_match":"密度","melisma_match":"拖腔","boundary_match":"起收音","range_match":"音域"};
    var tn = {"平":"平声","上":"上声","去":"去声","入":"入声"};
    var stl = "<table style=\"width:100%;border-collapse:collapse;font-size:12px\"><tr style=\"background:#fef3e0\"><th>指标</th><th>Bare</th><th>Rich</th><th>提升</th></tr>";
    var bSS = be ? be.style_scores : {};
    var rSS = re ? re.style_scores : {};
    for (var sk in sn) {
      var bv = bSS[sk]||0; var rv = rSS[sk]||0; var dv = cmp.style_delta ? (cmp.style_delta[sk]||0) : (rv-bv);
      stl += "<tr style=\"border-bottom:1px solid #eee\"><td>"+sn[sk]+"</td><td>"+pctC(bv)+"</td><td>"+pctC(rv)+"</td><td style=\"color:"+(dv>=0?"#2e7d32":"#c62828")+";font-weight:bold\">"+(dv>=0?"▲ ":"▼ ")+dStr(dv)+"</td></tr>";
    }
    var bTS = be ? be.tone_scores : {};
    var rTS = re ? re.tone_scores : {};
    for (var tk in tn) {
      var btv = (bTS[tk] != null && bTS[tk] >= 0) ? bTS[tk] : 0;
      var rtv = (rTS[tk] != null && rTS[tk] >= 0) ? rTS[tk] : 0;
      var dtv = cmp.tone_delta ? (cmp.tone_delta[tk]||0) : (rtv-btv);
      stl += "<tr style=\"border-bottom:1px solid #eee\"><td>声调-"+tn[tk]+"</td><td>"+pctC(btv)+"</td><td>"+pctC(rtv)+"</td><td style=\"color:"+(dtv>=0?"#2e7d32":"#c62828")+";font-weight:bold\">"+(dtv>=0?"▲ ":"▼ ")+dStr(dtv)+"</td></tr>";
    }
    var bSt = be ? be.style_overall : 0; var bTo = be ? be.tone_overall : 0; var bOv = be ? be.overall_score : 0;
    var rSt = re ? re.style_overall : 0; var rTo = re ? re.tone_overall : 0; var rOv = re ? re.overall_score : 0;
    stl += "<tr style=\"border-top:2px solid #d4a574;font-weight:bold;background:#fef9f0\"><td>风格综合</td><td>"+pctC(bSt)+"</td><td>"+pctC(rSt)+"</td><td style=\"color:"+((cmp.style_overall_delta||0)>=0?"#2e7d32":"#c62828")+"\">"+((cmp.style_overall_delta||0)>=0?"▲ ":"▼ ")+dStr(cmp.style_overall_delta)+"</td></tr>";
    stl += "<tr style=\"font-weight:bold;background:#fef9f0\"><td>声调综合</td><td>"+pctC(bTo)+"</td><td>"+pctC(rTo)+"</td><td style=\"color:"+((cmp.tone_overall_delta||0)>=0?"#2e7d32":"#c62828")+"\">"+((cmp.tone_overall_delta||0)>=0?"▲ ":"▼ ")+dStr(cmp.tone_overall_delta)+"</td></tr>";
    stl += "<tr style=\"font-weight:bold;background:#fef3e0;font-size:14px\"><td>综合评分</td><td>"+pctC(bOv)+"</td><td>"+pctC(rOv)+"</td><td style=\"color:"+((cmp.overall_delta||0)>=0?"#2e7d32":"#c62828")+"\">"+((cmp.overall_delta||0)>=0?"▲ ":"▼ ")+dStr(cmp.overall_delta)+"</td></tr>";
    stl += "</table>";
    document.getElementById("compare-delta-table").innerHTML = stl;
  }
  b.disabled = false;
  b.textContent = "🎵 生成音乐";
  return;
}
document.getElementById("compare-result-area").style.display = "none";
document.getElementById("trans-score-area").style.display = "none";
document.getElementById("result-area").style.display = "block";
document.getElementById("status-line").innerHTML = "Complete - "+d.tokens+" tokens - "+d.api_time+"s - "+d.n_groups+" groups";
document.getElementById("result-status").innerHTML =
  "<span>🟢 "+d.tokens+" tokens</span><span>⏱ "+d.api_time+"s</span><span>📐 "+d.n_groups+" 字音组</span><span>🎵 MusicXML已生成</span>" +
  (d.saved_xml_path ? "<br><span style=\"font-size:11px\">📁 已保存: "+escHtml(d.saved_xml_path)+"</span>" : "");
document.getElementById("result-area").style.display = "block";

var raw = d.raw_response || "";
document.getElementById("gongche-output").innerHTML =
  "<pre style=\"background:transparent;color:#5c2e0e;font-size:16px;line-height:2;letter-spacing:2px;max-height:500px;white-space:pre-wrap;padding:0;margin:0;\">" + escHtml(raw) + "</pre>";

document.getElementById("sys-prompt").textContent = d.system_prompt;
document.getElementById("usr-prompt").textContent = d.user_prompt;
document.getElementById("raw-resp").textContent = d.raw_response;
document.getElementById("feat-data").textContent = JSON.stringify(d.qupai_features,null,2);

if(d.musicxml_b64){
  var mx = document.getElementById("dl-mxl");
  mx.href = "data:application/vnd.recordare.musicxml+xml;base64,"+d.musicxml_b64;
  mx.download = q+"_"+l.replace(/\n/g,"").slice(0,8)+".musicxml";
}
var tx = document.getElementById("dl-txt");
tx.href = "data:text/plain;charset=utf-8,"+encodeURIComponent(d.raw_response);
tx.download = q+"_"+l.replace(/\n/g,"").slice(0,8)+".txt";
switchTab("gongche");

// ---- 渲染定量评估 ----
function renderEval(e, prefix) {
  prefix = prefix || "";
  var pct = function(v){ return (v != null && v >= 0) ? (v*100).toFixed(0)+"%" : "N/A"; };
  document.getElementById("eval-overall").textContent = pct(e.overall_score);
  document.getElementById("eval-style").textContent = pct(e.style_overall);
  document.getElementById("eval-tone").textContent = pct(e.tone_overall);
  document.getElementById("eval-tone-rule").textContent = pct(e.tone_rule_overall);
  document.getElementById("eval-tone-data").textContent = pct(e.tone_data_overall);
  document.getElementById("eval-meta").textContent = prefix + (e.timestamp || '') + " | " + (e.qupai || '') + " | " + (e.lyrics || '').slice(0,20);

  // 风格指标
  var styleNames = {"format_compliance":"格式合规","pitch_distribution":"音高分布匹配","interval_distribution":"音程分布匹配","density_match":"密度匹配","melisma_match":"拖腔匹配","boundary_match":"起收音匹配","range_match":"音域匹配"};
  var styleHtml = "";
  for(var k in styleNames){
    var v = (e.style_scores && e.style_scores[k] != null) ? e.style_scores[k] : 0;
    var p = (v*100).toFixed(0);
    styleHtml += '<div style="margin:3px 0"><span style="display:inline-block;width:110px">'+styleNames[k]+'</span>'+
      '<span style="display:inline-block;height:16px;background:linear-gradient(90deg,#8b4513,#d4a574);border-radius:3px;vertical-align:middle;width:'+(p*2)+'px;min-width:'+(p>0?'2px':'0')+'"></span>'+
      '<span style="font-size:11px;margin-left:6px">'+p+'%</span></div>';
  }
  document.getElementById("eval-style-detail").innerHTML = styleHtml;

  // 声调指标 — 综合
  var toneNames = {"平":"平声","上":"上声","去":"去声","入":"入声"};
  var toneHtml = "";
  for(var k in toneNames){
    var v = (e.tone_scores && e.tone_scores[k] != null) ? e.tone_scores[k] : null;
    if(v !== null && v >= 0){
      var p = (v*100).toFixed(0);
      toneHtml += '<div style="margin:3px 0"><span style="display:inline-block;width:110px">'+toneNames[k]+'</span>'+
        '<span style="display:inline-block;height:16px;background:linear-gradient(90deg,#1565c0,#42a5f5);border-radius:3px;vertical-align:middle;width:'+(p*2)+'px;min-width:2px"></span>'+
        '<span style="font-size:11px;margin-left:6px">'+p+'%</span></div>';
    } else {
      toneHtml += '<div style="margin:3px 0"><span style="display:inline-block;width:110px">'+toneNames[k]+'</span><span style="font-size:11px;color:#888">(未出现)</span></div>';
    }
  }
  document.getElementById("eval-tone-detail").innerHTML = toneHtml;

  // 声调指标 — 规则判断
  if(e.tone_rule_scores){
    var ruleHtml = '<p style="font-size:11px;color:#8b6914;margin-bottom:4px">方法A: 传统规则判断</p>';
    for(var k in toneNames){
      var v = (e.tone_rule_scores && e.tone_rule_scores[k] != null) ? e.tone_rule_scores[k] : null;
      if(v !== null && v >= 0){
        var p = (v*100).toFixed(0);
        ruleHtml += '<div style="margin:2px 0"><span style="display:inline-block;width:110px;font-size:12px">'+toneNames[k]+'</span>'+
          '<span style="display:inline-block;height:12px;background:linear-gradient(90deg,#ff9800,#ffc107);border-radius:2px;vertical-align:middle;width:'+(p*2)+'px;min-width:2px"></span>'+
          '<span style="font-size:10px;margin-left:4px">'+p+'%</span></div>';
      }
    }
    ruleHtml += '<span style="font-size:10px;color:#8b6914">综合: '+pct(e.tone_rule_overall)+'</span>';
    document.getElementById("eval-tone-rule-detail").innerHTML = ruleHtml;
  }

  // 声调指标 — 数据驱动
  if(e.tone_data_scores){
    var dataHtml = '<p style="font-size:11px;color:#1565c0;margin-bottom:4px">方法B: 数据集统计对比</p>';
    for(var k in toneNames){
      var v = (e.tone_data_scores && e.tone_data_scores[k] != null) ? e.tone_data_scores[k] : null;
      if(v !== null && v >= 0){
        var p = (v*100).toFixed(0);
        dataHtml += '<div style="margin:2px 0"><span style="display:inline-block;width:110px;font-size:12px">'+toneNames[k]+'</span>'+
          '<span style="display:inline-block;height:12px;background:linear-gradient(90deg,#1565c0,#64b5f6);border-radius:2px;vertical-align:middle;width:'+(p*2)+'px;min-width:2px"></span>'+
          '<span style="font-size:10px;margin-left:4px">'+p+'%</span></div>';
      }
    }
    dataHtml += '<span style="font-size:10px;color:#1565c0">综合: '+pct(e.tone_data_overall)+'</span>';
    document.getElementById("eval-tone-data-detail").innerHTML = dataHtml;
  }

  // 逐字分析
  if(e.char_details && e.char_details.length > 0){
    var charHtml = '<table style="width:100%;border-collapse:collapse">'+
      '<tr style="background:#fef3e0;font-size:11px"><th>字</th><th>声</th><th>音</th>'+
      '<th>稳</th><th>↕</th><th>短</th><th>首</th><th style="color:#e65100">规则</th>'+
      '<th style="color:#1565c0">JS</th><th style="color:#1565c0">首</th><th style="color:#1565c0">拖</th><th style="color:#1565c0">时</th>'+
      '<th style="color:#1565c0">数据</th><th>综合</th></tr>';
    for(var i=0;i<e.char_details.length;i++){
      var c = e.char_details[i];
      var arrow = (c.rule_contour||0) > 0.1 ? "↗" : (c.rule_contour||0) < -0.1 ? "↘" : "→";
      charHtml += '<tr style="border-bottom:1px solid #eee;font-size:10px">'+
        '<td style="font-weight:bold">'+escHtml(c.char)+'</td>'+
        '<td>'+c.tone+'</td>'+
        '<td>'+c.note_count+'</td>'+
        '<td>'+(c.rule_stability!=null?(c.rule_stability*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+arrow+'</td>'+
        '<td>'+(c.rule_brevity!=null?(c.rule_brevity*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+(c.rule_first_pitch!=null?(c.rule_first_pitch*100).toFixed(0)+"%":"-")+'</td>'+
        '<td style="color:#e65100;font-weight:bold">'+(c.rule_score!=null&&c.rule_score>=0?(c.rule_score*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+(c.data_pitch_js!=null?(c.data_pitch_js*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+(c.data_first_pitch!=null?(c.data_first_pitch*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+(c.data_melisma!=null?(c.data_melisma*100).toFixed(0)+"%":"-")+'</td>'+
        '<td>'+(c.data_duration!=null?(c.data_duration*100).toFixed(0)+"%":"-")+'</td>'+
        '<td style="color:#1565c0;font-weight:bold">'+(c.data_score!=null&&c.data_score>=0?(c.data_score*100).toFixed(0)+"%":"-")+'</td>'+
        '<td style="font-weight:bold;color:#6b3410">'+(c.score!=null&&c.score>=0?(c.score*100).toFixed(0)+"%":"-")+'</td></tr>';
    }
    charHtml += '</table>';
    document.getElementById("eval-char-detail").innerHTML = charHtml;
  } else {
    document.getElementById("eval-char-detail").innerHTML = '<span style="color:#888">暂无逐字分析数据</span>';
  }

  // 渲染指标计算说明
  if(e.metric_descriptions){
    var descHtml = '';
    var styleKeys = ['format_compliance','pitch_distribution','interval_distribution','density_match','melisma_match','boundary_match','range_match','style_overall'];
    var toneKeys = ['tone_rule_scores','tone_data_scores','tone_overall'];
    var allKeys = styleKeys.concat(toneKeys).concat(['overall_score']);
    allKeys.forEach(function(k){
      var d = e.metric_descriptions[k];
      if(d){
        descHtml += '<div style="margin:6px 0;padding:6px 8px;border-left:3px solid #d4a574;background:#fefdf8">';
        descHtml += '<b style="color:#6b3410">'+escHtml(d.label)+'</b>';
        descHtml += '<div style="color:#5c3a1e;margin-top:1px">'+escHtml(d.method)+'</div>';
        descHtml += '<div style="color:#8b6914;font-size:10px;margin-top:1px">' + 'Ref: '+escHtml(d.reference)+'</div>';
        descHtml += '<div style="color:#808080;font-size:10px;margin-top:1px">' + escHtml(d.meaning)+'</div>';
        descHtml += '</div>';
      }
    });
    document.getElementById("eval-method-detail").innerHTML = descHtml;
  }
}

if(d.evaluation){ renderEval(d.evaluation, ""); }
}catch(e){
document.getElementById("status-line").innerHTML = "❌ 请求失败";
alert("生成失败: "+e.message);
}
b.disabled = false;
b.textContent = "🎵 生成音乐";
}

function switchTab(n){
var tabs = document.querySelectorAll(".tab");
var names = ["gongche","evaluation","prompt","raw","features"];
tabs.forEach(function(t,i){t.classList.toggle("active",names[i]===n)});
document.querySelectorAll(".tab-content").forEach(function(t){t.classList.remove("active")});
document.getElementById("tab-"+n).classList.add("active");
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    @staticmethod
    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: Handler._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Handler._sanitize(v) for v in obj]
        return obj

    def _send_json(self, data, status=200):
        data = self._sanitize(data)
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(HTML_PAGE.replace("__QUPAI_DATA__", QUPAI_JSON).replace("__GONGS_DATA__", GONGS_JSON))
        elif self.path == "/api/gongs":
            self._send_json({"gongs": get_gongs()})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            try:
                body = raw_body.decode("utf-8")
            except UnicodeDecodeError:
                body = raw_body.decode("gbk", errors="replace")
            try:
                data = json.loads(body)
            except Exception:
                self._send_json({"error": "无效的 JSON 请求"}, 400)
                return

            engine = data.get("mode", "llm")  # "transformer" | "llm"
            gongdiao = data.get("gongdiao", "").strip()
            qupai = data.get("qupai", "").strip()
            lyrics = data.get("lyrics", "").strip()
            if not lyrics:
                self._send_json({"error": "请填写歌词"}, 400)
                return

            if engine == "transformer":
                if not gongdiao:
                    self._send_json({"error": "请选择宫调"}, 400)
                    return
                try:
                    title = data.get("title", "").strip() or "AI生成旋律"
                    result = generate_transformer(lyrics, gongdiao, title=title)
                except Exception as e:
                    self._send_json({"error": f"生成失败: {str(e)}"}, 500)
                    return
                eval_data = None
                if result["gc_groups"]:
                    eval_data = evaluate_generated(result["gc_groups"], lyrics, gongdiao)
                self._send_json({
                    "mode": "transformer",
                    "engine": "transformer",
                    "gongdiao": gongdiao,
                    "title": result.get("title", ""),
                    "items": result.get("items", []),
                    "raw_response": result["raw_response"],
                    "musicxml_b64": result["musicxml_b64"],
                    "saved_xml_path": result["saved_xml_path"],
                    "tokens": result["tokens"],
                    "api_time": result["api_time"],
                    "n_groups": result["n_groups"],
                    "evaluation": eval_data,
                })
                return

            # ---- LLM 模式（原有逻辑）----
            if not qupai:
                self._send_json({"error": "请提供曲牌名"}, 400)
                return
            if qupai not in FEATURES["qupai_features"]:
                self._send_json({"error": f"曲牌「{qupai}」不在数据集中"}, 400)
                return

            try:
                compare_mode = data.get("compare", False)
                if compare_mode:
                    bare_result = generate_bare(qupai, lyrics)
                    rich_result = generate(qupai, lyrics)

                    bare_eval = None
                    rich_eval = None
                    if bare_result["gc_groups"]:
                        bare_eval = evaluate_generated(bare_result["gc_groups"], lyrics, qupai)
                    if rich_result["gc_groups"]:
                        rich_eval = evaluate_generated(rich_result["gc_groups"], lyrics, qupai)

                    comparison = None
                    if bare_eval and rich_eval:
                        comparison = {
                            "style_delta": {
                                k: round(rich_eval["style_scores"].get(k, 0) - bare_eval["style_scores"].get(k, 0), 4)
                                for k in bare_eval["style_scores"]
                            },
                            "tone_delta": {
                                k: round(
                                    (rich_eval["tone_scores"].get(k, 0) if rich_eval["tone_scores"].get(k, -1) >= 0 else 0)
                                    - (bare_eval["tone_scores"].get(k, 0) if bare_eval["tone_scores"].get(k, -1) >= 0 else 0),
                                    4
                                )
                                for k in ["平", "上", "去", "入"]
                            },
                            "style_overall_delta": round(rich_eval["style_overall"] - bare_eval["style_overall"], 4),
                            "tone_overall_delta": round(rich_eval["tone_overall"] - bare_eval["tone_overall"], 4),
                            "overall_delta": round(rich_eval["overall_score"] - bare_eval["overall_score"], 4),
                        }

                    qf = FEATURES["qupai_features"].get(qupai, {})
                    self._send_json({
                        "mode": "compare",
                        "bare": {
                            "system_prompt": bare_result["system_prompt"],
                            "user_prompt": bare_result["user_prompt"],
                            "raw_response": bare_result["raw_response"],
                            "musicxml_b64": bare_result["musicxml_b64"],
                            "saved_xml_path": bare_result["saved_xml_path"],
                            "tokens": bare_result["tokens"],
                            "api_time": bare_result["api_time"],
                            "n_groups": bare_result["n_groups"],
                            "evaluation": bare_eval,
                        },
                        "rich": {
                            "system_prompt": rich_result["system_prompt"],
                            "user_prompt": rich_result["user_prompt"],
                            "raw_response": rich_result["raw_response"],
                            "musicxml_b64": rich_result["musicxml_b64"],
                            "saved_xml_path": rich_result["saved_xml_path"],
                            "tokens": rich_result["tokens"],
                            "api_time": rich_result["api_time"],
                            "n_groups": rich_result["n_groups"],
                            "evaluation": rich_eval,
                        },
                        "comparison": comparison,
                        "qupai_features": {
                            "name": qupai,
                            "song_count": qf.get("song_count", 0),
                            "start_pitches": qf.get("start_pitches", {}),
                            "end_pitches": qf.get("end_pitches", {}),
                            "pitch_stats": qf.get("pitch_stats", {}),
                            "density": qf.get("density", {}),
                            "region_distribution": qf.get("region_distribution", {}),
                            "mode_distribution": qf.get("mode_distribution", {}),
                        },
                    })
                else:
                    result = generate(qupai, lyrics)
            except Exception as e:
                self._send_json({"error": f"生成失败: {str(e)}"}, 500)
                return

            if not compare_mode:
                # 定量评估
                eval_data = None
                if result["gc_groups"]:
                    eval_data = evaluate_generated(
                        result["gc_groups"], lyrics, qupai
                    )

                qf = FEATURES["qupai_features"].get(qupai, {})
                self._send_json({
                "system_prompt": result["system_prompt"],
                "user_prompt": result["user_prompt"],
                "raw_response": result["raw_response"],
                "musicxml_b64": result["musicxml_b64"],
                "saved_xml_path": result["saved_xml_path"],
                "tokens": result["tokens"],
                "api_time": result["api_time"],
                "n_groups": result["n_groups"],
                "qupai_features": {
                    "name": qupai,
                    "song_count": qf.get("song_count", 0),
                    "start_pitches": qf.get("start_pitches", {}),
                    "end_pitches": qf.get("end_pitches", {}),
                    "pitch_stats": qf.get("pitch_stats", {}),
                    "density": qf.get("density", {}),
                    "region_distribution": qf.get("region_distribution", {}),
                    "mode_distribution": qf.get("mode_distribution", {}),
                },
                "evaluation": eval_data,
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    import threading, webbrowser
    port = int(os.getenv("PORT", "5000"))

    def open_browser():
        import time as _time
        _time.sleep(2)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"\n==========================================")
    print(f"  JiuGong Music Generation Web App")
    print(f"  Open: http://127.0.0.1:{port}")
    print(f"==========================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
