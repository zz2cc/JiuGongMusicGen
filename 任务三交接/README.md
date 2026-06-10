# 九宫旋律生成 · 交接说明

这是「输入歌词 → 选宫调 → 生成旋律 → 下载 MusicXML」项目的交接包，
包含可直接运行的网站、模型，以及完整的 notebook。

---

## 一、目录结构

```
计算机音乐_交接/
├── 九宫_整理版.ipynb                完整 notebook（数据处理 / 模型 / 训练 / 评估 / 生成 / 导出）
├── conditional_transformer_best.pt  训练好的模型（notebook 加载用）
├── 网站/
│   ├── app.py                       后端服务（加载模型 + 生成 + 导出）
│   ├── index.html                   前端页面
│   └── conditional_transformer_best.pt  模型（网站用，与根目录是同一个文件）
└── README.md                        本说明
```

> 没有附带 `JiuGongDataSet` 数据集和 `processed_songs.json`（约 400M）。
> 跑网站、加载模型评估都用不到它们；只有「从头预处理 / 重新训练 / 完整评估对比」才需要，
> 需要的话另外找我要。

---

## 二、环境配置

推荐用 Anaconda 新建一个环境（避免污染你现有环境）：

```bash
conda create -n jiugong python=3.10 -y
conda activate jiugong
```

安装依赖：

```bash
pip install torch music21 zhconv xpinyin pandas numpy
# 网站还需要：
pip install fastapi uvicorn
# 用 notebook 还需要：
pip install jupyter
```

> 没有 GPU 也能跑（模型不大，CPU 即可生成）。

---

## 三、跑网站（最简单，推荐先试这个）

```bash
cd 网站
python app.py
```

看到 `Uvicorn running on http://127.0.0.1:8000` 即启动成功。
浏览器打开 **http://127.0.0.1:8000** ，输入歌词、选宫调、点「生成曲谱」，
页面上会显示每个字对应的音，并可下载 MusicXML（可导入 MuseScore / Ace Studio）。

关闭：在命令行按 `Ctrl + C`。

常见问题：
- 报找不到模块 → 没激活环境，先 `conda activate jiugong`。
- 报找不到 `conditional_transformer_best.pt` → 模型要和 `app.py` 在同一目录（网站/ 下已附带）。
- 8000 端口被占用 → 改 `app.py` 最后一行的 `port=8000` 为其他值，浏览器地址同步改。

---

## 四、用 notebook

```bash
conda activate jiugong
jupyter notebook 九宫_整理版.ipynb
```

notebook 各节：环境 → 数据预处理 → 编码 → 模型 → 训练 → 生成 → 导出 → 对照模型 → 评估 → 对比实验。

**重点：想直接用现成模型做评估/生成，不必从头训练。**
1. 依次运行到「四、模型」，让模型类定义生效；
2. 运行「四之二、加载组员模型」这个 cell（会从 `.pt` 加载权重，模型结构 d_model=192 / num_layers=4）；
3. **跳过「五、训练」**，直接运行后面的生成、导出、评估。

> 注意：notebook 前几节（数据预处理、对照模型、真实分布统计）依赖 `JiuGongDataSet` 数据集，
> 本包未附带。只走「加载模型 → 生成 → 导出」这条线不需要数据集。
> 若要跑完整的评估对比（随机/马尔可夫基线、与真实语料的相似度），需要数据集。

---

## 五、模型说明

- 结构：条件式 Transformer，d_model=192，num_layers=4，nhead=8。
- 输入：宫调 + 逐字声调 + 是否字首；输出：每个字的音高、时值、是否结束。
- 随机种子已固定（SEED=42），相同输入可复现同一结果。
- 想换模型：替换 `.pt` 文件即可；若结构不同，同步改 `app.py` 第 2 节和 notebook 里建模型处的参数。
