"""
Prompt 模板工程模块。

定义 System Prompt 和 Few-Shot 样例格式，
用于引导 DeepSeek LLM 生成符合九宫大成风格的工尺谱旋律。

支持多种 prompt 策略变体用于消融实验。
"""

from typing import List, Dict, Optional, Tuple
from .gongche_vocab import format_lyric_gongche_pairs


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT_V1 = """你是一位精通中国古典音乐的大师，熟悉《九宫大成南北词宫谱》中的曲牌音乐体系。

## 工尺谱知识
工尺谱是中国传统记谱法，用汉字表示音高。标准工尺字符从低到高排列：
合(最低) → 四 → 一 → 上(宫音/主音) → 尺 → 工 → 凡 → 六 → 五 → 乙 → 仩(最高)

其中"上"为宫音（相当于西方音乐的do/主音）。常用五声音阶为：合-四-上-尺-工。

## 作曲规则
1. 一个歌词字可以对应1~3个工尺音符，称为"拖腔"。拖腔音符用空格分隔放在方括号内。
2. 旋律走向与歌词声调协调：
   - 平声字(阴平/阳平)多用上、尺等平稳音
   - 去声字常配合下行旋律（如五→六、工→尺→上）
   - 上声字常配合上行旋律（如上→尺→工）
   - 入声字常用短促的单音符
3. 同一曲牌的旋律在音高骨架、起音收音、板拍结构上具有相似性
4. 南词风格偏柔婉细腻，多用级进音程；北词风格偏刚健豪放，多用跳进音程
5. 乐曲起音和收音（首个字和末个字的主音）基本稳定在同一音高附近

## 节奏标记
在音符后添加时长标记：
- 无标记 = 四分音符（默认）
- (O) = 二分音符（长音）
- (x) = 八分音符
- (o) = 十六分音符

## 输出格式
对每句歌词，严格按以下格式输出旋律行（不要输出任何其他内容）：

旋律: <字1>[<工尺1> <工尺2>] <字2>[<工尺1> <工尺2>] ...

例如：
旋律: 風[工] 和[尺 工] 日[上 尺] 麗[工 五] 布[五 六] 艷[工] 陽[尺 工(O)]

请确保输出可以被直接解析。"""


SYSTEM_PROMPT_BRIEF = """你是中国古典音乐作曲大师。请根据曲牌风格和新歌词生成工尺谱旋律。
输出格式: 旋律: 字[工尺 工尺] 字[工尺] ...
拖腔(一字多音)用方括号内的空格分隔表示。
时长标记: 无=四分音符 (O)=二分 (x)=八分。
工尺音阶: 合 四 一 上 尺 工 凡 六 五 乙 仩 (上=宫音)
只输出旋律行，不要其他内容。"""


SYSTEM_PROMPT_DETAILED = """# 角色
你是一位精通中国古典音乐的AI作曲大师，专门为《九宫大成南北词宫谱》风格的曲牌音乐创作旋律。

# 背景知识
《九宫大成南北词宫谱》是清代乾隆年间编纂的中国古典音乐集大成之作，
收录超过6000首诗词音乐。记谱使用中国传统工尺谱系统。

# 工尺谱基础
工尺谱是中国传统记谱法，用汉字表示音高。字符从低到高：
```
合 → 四 → 一 → 上 → 尺 → 工 → 凡 → 六 → 五 → 乙 → 仩
```
其中"上"为宫音（主音），相当于C4（中央C）。
每个工尺字符之间的音程关系为：
- 相邻字符多为大二度（全音），如 上→尺、尺→工
- 特殊情况：凡→六 为小二度（半音）

# 作曲原理
## 依字行腔
旋律创作需遵循"依字行腔"原则，即旋律走向与歌词声调协调：
- 平声字（阴平、阳平）：旋律平稳，常用上、尺、工等核心音。阴平略高，阳平可略上行。
- 上声字：旋律多从较低音开始上行，如上→尺、四→上。
- 去声字：旋律多从较高音下行，如六→工、五→尺。
- 入声字：短促有力，多用单音符，不拖腔。

## 曲牌约束
- 同一曲牌的多首乐曲共享相似的旋律骨架（起音、收音、关键转折音）
- 板拍结构（每句的强拍位置）保持一致
- 拖腔（一字多音）的位置和长度有规律可循
- 南词风格细腻柔婉（级进为主），北词风格豪放刚健（跳进较多）

## 旋律结构
- 乐句通常以平稳音（上、尺）开始和结束
- 拖腔多出现在句尾字或情感关键字
- 音域一般在上下八度以内（合 到 仩）

# 输出格式
严格输出单行：
```
旋律: <字>[<工尺> <工尺>...] <字>[<工尺>...] ...
```

# 示例
输入曲牌《奉時春》，歌词"風和日麗布艷陽"：
旋律: 風[工] 和[尺 工] 日[上 尺] 麗[工 五] 布[五 六] 艷[工] 陽[尺 工(O)]

这表示：
- 風 → 一个工音符
- 和 → 尺 工 两个音符（拖腔）
- 陽 → 尺 工(O) 两个音符，最后是二分音符长音"""


# Prompt 变体注册表
SYSTEM_PROMPTS = {
    "v1": SYSTEM_PROMPT_V1,
    "brief": SYSTEM_PROMPT_BRIEF,
    "detailed": SYSTEM_PROMPT_DETAILED,
}


# ============================================================================
# Few-Shot 样例格式化
# ============================================================================


def format_fewshot_example(
    lyrics: str,
    lyric_note_groups: List,
    qupai: str = "",
    source: str = "",
    volume_region: str = "",
    volume_mode: str = "",
    use_markers: bool = True,
    include_stats: bool = True,
) -> str:
    """
    格式化单个 Few-Shot 样例。

    Returns:
        ```
        ## 示例：曲牌《奉時春》（南詞·仙呂宮引）
        来源: 月令承應
        歌词: 風和日麗布艷陽
        旋律: 風[工] 和[尺 工] 日[上 尺] ...
        密度: 1.71音/字
        ```
    """
    parts = []

    # 标题
    meta = []
    if qupai:
        meta.append(f"《{qupai}》")
    if volume_region or volume_mode:
        region_mode = f"{volume_region}·{volume_mode}" if volume_region and volume_mode else (volume_region or volume_mode)
        meta.append(region_mode)
    if meta:
        parts.append(f"## 示例：曲牌{'·'.join(meta)}")

    # 来源
    if source:
        parts.append(f"来源: {source}")

    # 歌词 & 旋律
    if lyrics:
        parts.append(f"歌词: {lyrics}")

    gongche_text = format_lyric_gongche_pairs(lyric_note_groups, use_markers=use_markers)
    parts.append(f"旋律: {gongche_text}")

    # 统计
    if include_stats:
        total_notes = sum(len(g.notes) for g in lyric_note_groups)
        total_lyrics = len(lyric_note_groups)
        density = total_notes / max(total_lyrics, 1)
        parts.append(f"密度: {density:.2f}音/字")

    return "\n".join(parts)


def format_fewshot_examples(
    examples: List[Dict],
    max_examples: int = 3,
    use_markers: bool = True,
    annotation: str = "full",  # "full", "minimal", "melody_only"
) -> str:
    """
    格式化多个 Few-Shot 样例。

    Args:
        examples: 从 QupaiIndex.retrieve_examples() 返回的样例列表
        max_examples: 最大样例数
        use_markers: 是否使用时长标记
        annotation: 标注模式
            - "full": 完整标注 (曲牌、来源、歌词、旋律、统计)
            - "minimal": 最简标注 (仅歌词→旋律)
            - "melody_only": 仅旋律

    Returns:
        格式化的 Few-Shot 文本块
    """
    if not examples:
        return ""

    parts = ["# Few-Shot 参考样例\n"]
    for i, ex in enumerate(examples[:max_examples]):
        if annotation == "minimal":
            text = f"歌词: {ex['lyrics']}"
            gongche = format_lyric_gongche_pairs(
                ex["lyric_note_groups"], use_markers=use_markers
            )
            text += f"\n旋律: {gongche}"
        elif annotation == "melody_only":
            text = format_lyric_gongche_pairs(
                ex["lyric_note_groups"], use_markers=use_markers
            )
        else:
            text = format_fewshot_example(
                lyrics=ex["lyrics"],
                lyric_note_groups=ex["lyric_note_groups"],
                qupai=ex.get("qupai", ""),
                source=ex.get("source", ""),
                volume_region=ex.get("volume_region", ""),
                volume_mode=ex.get("volume_mode", ""),
                use_markers=use_markers,
                include_stats=True,
            )

        parts.append(text)
        if i < min(len(examples), max_examples) - 1:
            parts.append("")

    return "\n".join(parts)


# ============================================================================
# 任务 Prompt 构建
# ============================================================================


def build_task_prompt(
    qupai: str,
    lyrics: str,
    qupai_info: Optional[Dict] = None,
    style_hint: Optional[str] = None,
) -> str:
    """
    构建当前生成任务 Prompt。

    Args:
        qupai: 曲牌名
        lyrics: 目标歌词
        qupai_info: 曲牌元信息
        style_hint: 风格提示
    """
    parts = ["\n# 生成任务\n"]

    parts.append(f"曲牌: 《{qupai}》")

    if qupai_info:
        if qupai_info.get("volume_region"):
            parts.append(f"板块: {qupai_info['volume_region']}")
        if qupai_info.get("volume_mode"):
            parts.append(f"宫调: {qupai_info['volume_mode']}")
        if qupai_info.get("source"):
            parts.append(f"来源风格: {qupai_info['source']}")
        if qupai_info.get("avg_notes_per_lyric"):
            parts.append(f"参考密度: {qupai_info['avg_notes_per_lyric']:.2f}音/字")

    if style_hint:
        parts.append(f"风格要求: {style_hint}")

    parts.append(f"\n新歌词: {lyrics}")
    parts.append("\n请按照参考样例的风格，为以上新歌词生成工尺谱旋律。")
    parts.append("请只输出一行 '旋律: ...'，不要解释。")

    return "\n".join(parts)


def build_full_prompt(
    fewshot_text: str,
    task_text: str,
) -> str:
    """合并 few-shot 样例和任务描述为用户消息"""
    parts = []
    if fewshot_text:
        parts.append(fewshot_text)
    if task_text:
        parts.append(task_text)
    return "\n".join(parts)


# ============================================================================
# Prompt 构建器
# ============================================================================


class PromptBuilder:
    """
    Prompt 构建器，支持多种策略变体用于实验。

    Usage:
        builder = PromptBuilder(strategy="full")
        system, user = builder.build(qupai="奉時春", lyrics="...", examples=[...])
    """

    def __init__(
        self,
        strategy: str = "full",
        system_prompt_version: str = "v1",
        use_markers: bool = True,
        annotation: str = "full",
    ):
        """
        Args:
            strategy: "zeroshot", "fewshot", "fewshot_tones", "fewshot_skeleton"
            system_prompt_version: "v1", "brief", "detailed"
            use_markers: 是否使用时长标记
            annotation: "full", "minimal", "melody_only"
        """
        self.strategy = strategy
        self.system_prompt = SYSTEM_PROMPTS.get(system_prompt_version, SYSTEM_PROMPT_V1)
        self.use_markers = use_markers
        self.annotation = annotation

    def build(
        self,
        qupai: str,
        lyrics: str,
        examples: Optional[List[Dict]] = None,
        qupai_info: Optional[Dict] = None,
        style_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        构建完整的 System + User prompt。

        Returns:
            (system_prompt, user_prompt)
        """
        system = self.system_prompt

        # 根据策略变体修改 system prompt
        if self.strategy == "fewshot_tones":
            system += "\n\n特别注意：请严格按照每字的声调来安排旋律走向。以下提供每个新歌词字的声调信息。"

        if self.strategy == "fewshot_skeleton":
            system += "\n\n特别注意：请严格参考样例中该曲牌的旋律骨架（起音→发展→收音）。"

        # 构建用户消息
        if self.strategy == "zeroshot":
            # 零样本：不提供样例
            user = build_task_prompt(qupai, lyrics, qupai_info, style_hint)
        else:
            # 构建 Few-Shot 部分
            if examples:
                fewshot = format_fewshot_examples(
                    examples,
                    max_examples=3,
                    use_markers=self.use_markers,
                    annotation=self.annotation,
                )
            else:
                fewshot = ""

            # 构建任务部分
            task = build_task_prompt(qupai, lyrics, qupai_info, style_hint)

            # 对于带声调的变体，添加声调信息
            if self.strategy == "fewshot_tones" and lyrics:
                tone_hint = _build_tone_hint(lyrics)
                task += f"\n\n{tone_hint}"

            # 对于骨架变体，添加骨架信息
            if self.strategy == "fewshot_skeleton" and qupai_info:
                skeleton_hint = _build_skeleton_hint(qupai_info)
                if skeleton_hint:
                    task += f"\n\n{skeleton_hint}"

            user = build_full_prompt(fewshot, task)

        return system, user


def _build_tone_hint(lyrics: str) -> str:
    """为每个字添加声调参考信息"""
    # 这里不做实际查字典，给一个占位提示让LLM自己判断
    chars = list(lyrics)
    return f"歌词共{len(chars)}字。请根据每字的平上去入声调安排旋律走向。"


def _build_skeleton_hint(qupai_info: Dict) -> str:
    """构建旋律骨架提示"""
    hints = []
    if qupai_info.get("common_start_gongche"):
        hints.append(f"起音常用工尺: {qupai_info['common_start_gongche']}")
    if qupai_info.get("common_end_gongche"):
        hints.append(f"收音常用工尺: {qupai_info['common_end_gongche']}")
    if qupai_info.get("pitch_range"):
        lo, hi = qupai_info["pitch_range"]
        hints.append(f"常用音域: {lo:.0f} ~ {hi:.0f}")
    if qupai_info.get("avg_notes_per_lyric"):
        hints.append(f"平均密度: {qupai_info['avg_notes_per_lyric']:.1f}音/字")

    if hints:
        return "该曲牌关键信息：\n" + "\n".join(hints)
    return ""


# ============================================================================
# 实验用 Prompt 变体
# ============================================================================


def get_experiment_prompts() -> Dict[str, dict]:
    """
    获取所有实验用 Prompt 变体配置。

    Returns:
        {
            "zeroshot": {"strategy": "zeroshot", ...},
            "fewshot_minimal": {"strategy": "fewshot", "annotation": "minimal", ...},
            ...
        }
    """
    return {
        "zeroshot": {
            "strategy": "zeroshot",
            "system_prompt_version": "v1",
            "annotation": "full",
        },
        "fewshot_minimal": {
            "strategy": "fewshot",
            "system_prompt_version": "v1",
            "annotation": "minimal",
        },
        "fewshot_full": {
            "strategy": "fewshot",
            "system_prompt_version": "v1",
            "annotation": "full",
        },
        "fewshot_tones": {
            "strategy": "fewshot_tones",
            "system_prompt_version": "detailed",
            "annotation": "full",
        },
        "fewshot_skeleton": {
            "strategy": "fewshot_skeleton",
            "system_prompt_version": "detailed",
            "annotation": "full",
        },
    }
