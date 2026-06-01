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
├── run_rich_generate.py      # 命令行生成入口
├── 启动.bat                   # Windows 一键启动
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py         # CSV 数据加载
│   ├── gongche_vocab.py       # 工尺谱词表、编码、解析
│   ├── musicxml_writer.py     # 工尺谱 → MusicXML 转换
│   ├── qupai_index.py         # 曲牌索引与 Few-Shot 检索
│   ├── prompt_templates.py    # Prompt 模板工程
│   ├── deepseek_client.py     # DeepSeek API 客户端
│   ├── generator.py           # 主生成管线
│   ├── evaluator.py           # 评估系统
│   ├── feature_extractor.py   # 特征提取器
│   ├── rich_prompt.py         # Rich Prompt 构建器
│   └── config.py              # 配置文件
├── features_cache.json        # 特征缓存（预计算，3.6 MB）
├── features_cache.pkl         # 特征缓存（二进制，加载更快）
├── FINAL_SONGS.csv            # 歌曲元数据 (6,563首)
├── FINAL_NOTES.csv            # 音符数据 (~696K行)
├── FINAL_BEATS.csv            # 板拍数据 (~296K行)
├── musicxml/                  # 原始 MusicXML 文件 (6,563个)
└── experiments/
    └── outputs/               # 生成输出（.txt + .musicxml）
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

| 指标 | 说明 |
|------|------|
| 格式合规率 | 输出是否可解析为有效工尺谱 |
| 音高覆盖率 | 是否使用了该曲牌的典型音高 |
| 音域匹配 | 生成旋律的音域是否合理 |
| 板拍密度 | 每字对应音符数是否合理 |
| 起音/收音 | 首音与末音是否符合曲牌惯例 |
| 音程分布 | 音程跳跃模式是否匹配 |

## 数据集

《九宫大成南北词宫谱》数据集：
- **6,563** 首诗词音乐
- **696,216** 个音符事件
- **296,055** 个板拍
- **2,053** 个曲牌
- **73** 种宫调
- **14** 个工尺谱字符

数据来源：香港理工大学《九宫大成》数据库
