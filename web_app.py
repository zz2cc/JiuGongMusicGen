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
from src.gongche_vocab import parse_compact_gongche
from src.musicxml_writer import gongche_to_musicxml
from src.data_loader import get_dataset as _get_ds

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
print(f"Ready: {len(QUPAI_DATA)} qupai")


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
        try:
            gongche_to_musicxml(lyric_groups=gc_groups, output_path=saved_xml_path,
                                title=f"AI: {qupai}", work_title=f"JiuGong: {qupai}", qupai=qupai)
            with open(saved_xml_path, "rb") as f:
                mxl_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            saved_xml_path = ""

    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "raw_response": raw,
            "musicxml_b64": mxl_b64, "saved_xml_path": saved_xml_path,
            "tokens": usage.get("total_tokens", 0),
            "api_time": round(api_time, 2), "n_groups": len(gc_groups)}


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
<label>曲牌名 <span style="font-weight:normal;color:#a08060">（输入筛选，仅可选数据集中已有的）</span></label>
<input type="text" id="qupai-filter" placeholder="输入拼音或汉字筛选曲牌..." autocomplete="off" oninput="filterQupai()">
<select id="qupai" size="10" onchange="onQupaiSelect()"></select>
<div style="font-size:11px;color:#8b6914;margin-top:2px" id="qupai-info"></div>

<label>歌词内容（自动按标点换行）</label>
<textarea id="lyrics" placeholder="小院春寒，梨花半落胭脂雨。绿窗朱户，燕绕秋千柱。" oninput="previewLines()">小院春寒，梨花半落胭脂雨。绿窗朱户，燕绕秋千柱。心事轻梳，欲语还羞住。风起处，落红无数，谁共斜阳暮？</textarea>
<div class="preview-lines" id="line-preview"></div>

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

<script>
var QDATA = __QUPAI_DATA__;
var sel = document.getElementById("qupai");
var filterInput = document.getElementById("qupai-filter");
var infoDiv = document.getElementById("qupai-info");

// Populate select
function populateSelect(list) {
  sel.innerHTML = "";
  list.forEach(function(q, i){
    var o = document.createElement("option");
    o.value = q.name;
    o.textContent = q.name + "  [" + q.count + "首 " + q.region + (q.mode?" "+q.mode:"") + "]";
    if (i === 0) o.selected = true;
    sel.appendChild(o);
  });
  updateInfo();
}
populateSelect(QDATA);

function filterQupai() {
  var val = filterInput.value.trim().toLowerCase();
  if (!val) { populateSelect(QDATA); return; }
  var filtered = QDATA.filter(function(q){
    return q.name.toLowerCase().indexOf(val) >= 0;
  });
  populateSelect(filtered);
}

function onQupaiSelect() {
  updateInfo();
}

function updateInfo() {
  var v = sel.value;
  var found = QDATA.filter(function(q){ return q.name === v; })[0];
  if (found) {
    infoDiv.textContent = "已选: " + found.name + " — 数据集有 " + found.count + " 首参考曲目";
  } else {
    infoDiv.textContent = "";
  }
}
updateInfo();

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

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
var q = sel.value.trim();
if (!q) { alert("请先在列表中选择一个曲牌"); return; }
var rawText = document.getElementById("lyrics").value.trim();
var lines = splitLyrics(rawText);
var l = lines.join("\n");
if(!q||!l){alert("请填写歌词");return}

var b = document.getElementById("gen-btn");
b.disabled = true;
b.textContent = "生成中...";
document.getElementById("status-line").innerHTML = "<span class='loading'>⏳ 正在生成...</span>";

try{
var r = await fetch("/api/generate",{
  method:"POST",
  headers:{"Content-Type":"application/json"},
  body:JSON.stringify({qupai:q, lyrics:l})
});

if(!r.ok){alert("HTTP "+r.status+": 服务异常，请稍后重试");b.disabled=false;b.textContent="🎵 生成音乐";return}

var d = await r.json();
if(d.error){alert(d.error);b.disabled=false;b.textContent="🎵 生成音乐";return}

lastResult = d;
document.getElementById("status-line").innerHTML = "✅ 完成 · "+d.tokens+" tokens · "+d.api_time+"s · "+d.n_groups+" 字音组";
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
}catch(e){
document.getElementById("status-line").innerHTML = "❌ 请求失败";
alert("生成失败: "+e.message);
}
b.disabled = false;
b.textContent = "🎵 生成音乐";
}

function switchTab(n){
var tabs = document.querySelectorAll(".tab");
var names = ["gongche","prompt","raw","features"];
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
            self._send_html(HTML_PAGE.replace("__QUPAI_DATA__", QUPAI_JSON))
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

            qupai = data.get("qupai", "").strip()
            lyrics = data.get("lyrics", "").strip()
            if not qupai or not lyrics:
                self._send_json({"error": "请提供曲牌名和歌词"}, 400)
                return
            if qupai not in FEATURES["qupai_features"]:
                self._send_json({"error": f"曲牌「{qupai}」不在数据集中"}, 400)
                return

            try:
                result = generate(qupai, lyrics)
            except Exception as e:
                self._send_json({"error": f"生成失败: {str(e)}"}, 500)
                return

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
