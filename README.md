# 九宫大成诗词音乐生成系统

基于《九宫大成南北词宫谱》数据集（6,563首曲目），使用 DeepSeek 大语言模型进行中国古典诗词音乐的自动生成。

**输入**：曲牌名 + 歌词文本  
**输出**：工尺谱文本（`.txt`）+ MusicXML 五线谱（`.musicxml`，可用 MuseScore 打开播放）

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```bash
cp .env.example .env
# 编辑 .env，替换 sk-xxxx 为你的真实 Key
```

### 3. 启动 Web 界面

```bash
python web_app.py
```

浏览器自动打开 `http://127.0.0.1:5000`。

Windows 用户可直接双击 `启动.bat`。

### 4. （可选）重新构建特征缓存

如果修改了 `feature_extractor.py` 或新增了数据：

```bash
python build_features.py
```

---

## 项目结构

```
JiuGongDataSet/
├── web_app.py                # Web 界面入口
├── build_features.py         # 特征缓存构建（一次性）
├── run_rich_generate.py      # Rich Prompt 生成
├── run_generate.py           # 基础生成
├── 启动.bat                   # Windows 一键启动
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py         # CSV 数据加载
│   ├── gongche_vocab.py       # 工尺谱词表、编码、解析
│   ├── musicxml_writer.py     # 工尺谱 → MusicXML 转换
│   ├── qupai_index.py         # 曲牌索引与 Few-Shot 检索
│   ├── prompt_templates.py    # Prompt 模板（含 Bare/Rich）
│   ├── deepseek_client.py     # DeepSeek API 客户端
│   ├── generator.py           # 主生成管线
│   ├── evaluator.py           # 原有评估器
│   ├── feature_extractor.py   # 特征提取器
│   ├── rich_prompt.py         # Rich Prompt 构建器
│   ├── quantitative_eval.py   # 定量评估系统
│   ├── tone_aligner.py        # 声调标注
│   └── config.py              # 配置文件
├── features_cache.json        # 预计算特征缓存 (3.6 MB)
├── FINAL_SONGS.csv            # 歌曲元数据 (6,563首)
├── FINAL_NOTES.csv            # 音符数据 (~696K行)
├── FINAL_BEATS.csv            # 板拍数据 (~296K行)
├── musicxml/                  # 原始 MusicXML 文件 (6,563个)
└── experiments/
    └── outputs/               # 生成输出（.txt + .musicxml + .eval.json）
```

## 工尺谱输出格式

```
曲牌: 奉時春
板块: 南詞
宫调: 仙呂宮引

歌词: 風和日麗布艷陽
旋律: 風[工] 和[尺 工] 日[上 尺] 麗[工 五] 布[五 六] 艷[工] 陽[尺 工(O)]
工尺序列: 工 尺 工 上 尺 工 五 五 六 工 尺 工(O)
密度: 1.71音/字 (12音/7字)
```

格式说明：
- `<字>[<工尺> <工尺>...]` 表示该字对应的工尺音符序列
- 方括号内多个音符 = 拖腔（一字多音）
- `(O)` = 二分音符长音，`(x)` = 八分音符，默认 = 四分音符
- 工尺音阶（低→高）：合 四 一 上 尺 工 凡 六 五 乙 仩 伬 仜

## 评估指标

系统对每次生成进行**定量评估**，分为两大维度：

### 风格相似度（70% 权重）

| 指标 | 说明 | 参考基准 |
|------|------|----------|
| 格式合规率 | 输出是否可解析为有效工尺谱 | GONGCHE_TO_PITCH 字典 |
| 音高分布匹配 | 工尺字使用频次 vs 曲牌数据集分布 | qupai_features.pitch_distribution |
| 音程分布匹配 | 音程跳跃模式余弦相似度 | qupai_features.interval_distribution |
| 密度匹配 | 每字音符数 vs 曲牌均值 | qupai_features.density |
| 拖腔匹配 | 拖腔率 + 句中位置分布 | qupai_features.melisma |
| 起收音匹配 | 首音/末音 ±3 半音邻近匹配 | qupai_features.start/end_pitches |
| 音域匹配 | 音高跨度比值 | qupai_features.pitch_stats |

### 声调对齐度（30% 权重）

| 方法 | 说明 | 参考基准 |
|------|------|----------|
| 规则判断 | 依字行腔规则：平声稳、上声升、去声降、入声短 | 传统曲唱理论 |
| 数据驱动 | 按声调聚合后与数据集 69 万音符统计对比 | tone_features.guangyun_tones |

### 对比模式

勾选「对比模式」复选框，系统同时用**极简提示词（Bare）**和**提示词工程（Rich）**各生成一次，右侧展示 Rich 完整评估报告，下方展示 Bare 工尺谱 + 逐项提升差异表，直观量化 Prompt Engineering 的效果。

### 句间休止

生成的 MusicXML 在每句歌词之间自动插入小幅休止符（八分音符时值），使播放时有自然的句读停顿。


## 数据集

《九宫大成南北词宫谱》数据集：
- **6,563** 首诗词音乐
- **696,216** 个音符事件
- **296,055** 个板拍
- **2,053** 个曲牌
- **73** 种宫调
- **14** 个工尺谱字符

数据来源：香港理工大学《九宫大成》数据库

## License

MIT

