"""
主生成管线模块 — 串联检索、Prompt构建、API调用、输出解析，产生双格式输出。

核心类: QupaiMusicGenerator
- 输入: 曲牌 + 歌词
- 输出: 工尺谱文本 (.txt) + MusicXML (.musicxml) + 元数据 (.meta.json)
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import Config
from .data_loader import JiuGongDataset, get_dataset, LyricNoteGroup
from .gongche_vocab import (
    parse_llm_response,
)
from .musicxml_writer import gongche_to_musicxml
from .qupai_index import QupaiIndex
from .prompt_templates import PromptBuilder, get_experiment_prompts
from .deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


@dataclass
class GeneratedPiece:
    """生成结果"""
    qupai: str
    lyrics: str
    melody: List[LyricNoteGroup]     # 解析后的工尺谱旋律
    gongche_text: str                # 工尺谱文本格式
    musicxml_path: str               # MusicXML 文件路径
    meta_path: str                   # 元数据 JSON 路径
    txt_path: str                    # 文本输出路径
    raw_response: str                # API 原始响应
    token_usage: dict                # Token 用量
    examples_used: List[Dict]        # 使用的 few-shot 样例
    generation_time: float           # 生成耗时 (秒)
    success: bool = True
    error_message: str = ""


class QupaiMusicGenerator:
    """
    九宫大成曲牌音乐生成器。

    Usage:
        config = Config.from_env()
        ds = get_dataset()
        gen = QupaiMusicGenerator(config, ds)
        gen.initialize()

        piece = gen.generate("奉時春", "風和日麗布艷陽")
        print(piece.gongche_text)
        print(piece.musicxml_path)  # 可在 MuseScore 打开

        results = gen.run_experiment("消融实验", tasks, variants)
    """

    def __init__(self, config: Config, dataset: Optional[JiuGongDataset] = None):
        self.config = config
        self.dataset = dataset
        self.index: Optional[QupaiIndex] = None
        self.client: Optional[DeepSeekClient] = None
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, build_index: bool = True):
        """初始化（加载数据、构建索引、连接 API）"""
        if self.dataset is None:
            logger.info("加载数据集...")
            cache_path = str(self.config.data_dir / ".jiugong_cache.pkl")
            self.dataset = get_dataset(
                data_dir=self.config.data_dir,
                cache_path=cache_path if Path(cache_path).exists() else None,
            )

        if build_index and self.index is None:
            self.index = QupaiIndex(self.dataset)
            self.index.build(min_song_count=1)
            logger.info(f"曲牌索引: {len(self.index)} 个曲牌")

        if self.client is None:
            self.client = DeepSeekClient(self.config.deepseek)
            logger.info("DeepSeek 客户端已初始化")

    def generate(
        self,
        qupai: str,
        lyrics: str,
        region: Optional[str] = None,
        mode: Optional[str] = None,
        style_hint: Optional[str] = None,
        n_examples: Optional[int] = None,
        prompt_strategy: str = "fewshot",
        temperature: Optional[float] = None,
        output_musicxml: bool = True,
        output_dir: Optional[str] = None,
    ) -> GeneratedPiece:
        """
        输入曲牌+歌词，生成工尺谱旋律。

        Args:
            qupai: 曲牌名（如 "奉時春"）
            lyrics: 歌词文本（如 "風和日麗布艷陽"）
            region: 南词/北词/合套（可选，不传则从曲牌推断）
            mode: 宫调（可选）
            style_hint: 风格提示（可选，如 "欢快活泼"）
            n_examples: few-shot 样例数（默认用配置值）
            prompt_strategy: prompt 策略（zeroshot/fewshot/fewshot_tones/fewshot_skeleton）
            temperature: 温度参数
            output_musicxml: 是否输出 MusicXML
            output_dir: 输出目录（覆盖配置中的目录）

        Returns:
            GeneratedPiece
        """
        if self.index is None:
            self.initialize()

        t0 = time.time()
        n_ex = n_examples or self.config.generator.n_fewshot_examples

        # ---- 1. 检索 Few-Shot 样例 ----
        examples = []
        if prompt_strategy != "zeroshot":
            examples = self.index.retrieve_examples(qupai, lyrics, n=n_ex)
            logger.info(f"检索到 {len(examples)} 个 Few-Shot 样例")

        # ---- 2. 获取曲牌元信息 ----
        qupai_info = {}
        profile = self.index.get_profile(qupai.replace("《", "").replace("》", ""))
        if profile:
            qupai_info = {
                "volume_region": region or (
                    max(profile.regions, key=profile.regions.get)
                    if profile.regions else ""
                ),
                "volume_mode": mode or (
                    max(profile.modes, key=profile.modes.get)
                    if profile.modes else ""
                ),
                "source": (
                    max(profile.sources, key=profile.sources.get)
                    if profile.sources else ""
                ),
                "avg_notes_per_lyric": profile.avg_notes_per_lyric,
                "avg_melisma_ratio": profile.avg_melisma_ratio,
                "pitch_range": profile.avg_pitch_range,
            }
            if profile.common_start_pitches:
                qupai_info["common_start_gongche"] = list(
                    profile.common_start_pitches.keys()
                )[:3]
            if profile.common_end_pitches:
                qupai_info["common_end_gongche"] = list(
                    profile.common_end_pitches.keys()
                )[:3]

        # 用传入的 region/mode 覆盖
        if region:
            qupai_info["volume_region"] = region
        if mode:
            qupai_info["volume_mode"] = mode

        # ---- 3. 构建 Prompt ----
        sys_version = "detailed" if prompt_strategy in ("fewshot_tones", "fewshot_skeleton") else "v1"
        builder = PromptBuilder(
            strategy=prompt_strategy,
            system_prompt_version=sys_version,
            use_markers=True,
            annotation="full" if prompt_strategy != "zeroshot" else "minimal",
        )

        system_prompt, user_prompt = builder.build(
            qupai=qupai,
            lyrics=lyrics,
            examples=examples,
            qupai_info=qupai_info,
            style_hint=style_hint,
        )

        # ---- 4. 调用 API ----
        logger.info(f"调用 DeepSeek API (模型: {self.config.deepseek.model})...")
        try:
            raw_text, usage = self.client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            logger.info(f"  响应长度: {len(raw_text)} 字符, "
                        f"Tokens: {usage['total_tokens']}")
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return GeneratedPiece(
                qupai=qupai, lyrics=lyrics,
                melody=[], gongche_text="",
                musicxml_path="", meta_path="", txt_path="",
                raw_response="", token_usage={},
                examples_used=examples,
                generation_time=time.time() - t0,
                success=False, error_message=str(e),
            )

        # ---- 5. 解析输出 ----
        melody_groups, gongche_text = parse_llm_response(raw_text)

        # ---- 6. 确定输出路径 ----
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        qupai_safe = qupai.replace("《", "").replace("》", "")
        file_hash = hashlib.md5((qupai_safe + lyrics).encode()).hexdigest()[:8]
        base_name = f"{qupai_safe}_{file_hash}"

        # ---- 7. 保存工尺谱文本 ----
        txt_path = out_dir / f"{base_name}.txt"
        txt_content = self._format_output_text(
            qupai, lyrics, gongche_text, melody_groups,
            qupai_info, examples, usage
        )
        txt_path.write_text(txt_content, encoding="utf-8")
        logger.info(f"工尺谱文本已保存: {txt_path}")

        # ---- 8. 转换为 MusicXML ----
        xml_path = ""
        if output_musicxml and melody_groups:
            xml_path_out = out_dir / f"{base_name}.musicxml"
            try:
                title = f"{qupai_safe}:{lyrics[:10]}"
                gongche_to_musicxml(
                    lyric_groups=melody_groups,
                    output_path=str(xml_path_out),
                    title=title,
                    work_title=f"AI生成·{qupai_safe}",
                    qupai=qupai_safe,
                )
                xml_path = str(xml_path_out)
                logger.info(f"MusicXML 已保存: {xml_path}")
            except Exception as e:
                logger.error(f"MusicXML 转换失败: {e}")

        # ---- 9. 保存元数据 ----
        meta_path = out_dir / f"{base_name}.meta.json"
        meta = {
            "qupai": qupai_safe,
            "lyrics": lyrics,
            "gongche_text": gongche_text,
            "musicxml_path": xml_path,
            "txt_path": str(txt_path),
            "prompt_strategy": prompt_strategy,
            "examples_used": [ex.get("polyu_id", "") for ex in examples],
            "token_usage": usage,
            "generation_time": round(time.time() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        logger.info(f"生成完成，耗时 {elapsed:.1f}s")
        if melody_groups:
            total_notes = sum(len(g.notes) for g in melody_groups)
            logger.info(f"  旋律: {len(melody_groups)} 字, {total_notes} 音, "
                        f"密度: {total_notes / max(len(melody_groups), 1):.2f}")

        return GeneratedPiece(
            qupai=qupai_safe,
            lyrics=lyrics,
            melody=melody_groups,
            gongche_text=gongche_text,
            musicxml_path=xml_path,
            meta_path=str(meta_path),
            txt_path=str(txt_path),
            raw_response=raw_text,
            token_usage=usage,
            examples_used=examples,
            generation_time=elapsed,
            success=bool(melody_groups),
            error_message="" if melody_groups else "无法解析生成的旋律",
        )

    def generate_with_variants(
        self,
        qupai: str,
        lyrics: str,
        n_variants: int = 3,
        temperature_range: Tuple[float, float] = (0.7, 1.0),
        **kwargs,
    ) -> List[GeneratedPiece]:
        """
        使用不同温度参数生成多个旋律变体。

        Returns:
            List[GeneratedPiece] — 每个变体都有 .txt + .musicxml
        """
        results = []
        temps = [
            temperature_range[0] + i * (temperature_range[1] - temperature_range[0]) / max(n_variants - 1, 1)
            for i in range(n_variants)
        ]

        for i, temp in enumerate(temps):
            logger.info(f"生成变体 {i + 1}/{n_variants} (temperature={temp:.2f})")
            piece = self.generate(
                qupai=qupai, lyrics=lyrics,
                temperature=temp, **kwargs,
            )
            results.append(piece)

        return results

    def batch_generate(
        self,
        tasks: List[Dict],
        **kwargs,
    ) -> List[GeneratedPiece]:
        """
        批量生成。

        Args:
            tasks: [{"qupai": "奉時春", "lyrics": "..."}, ...]

        Returns:
            List[GeneratedPiece]
        """
        results = []
        for i, task in enumerate(tasks):
            logger.info(f"批量生成 {i + 1}/{len(tasks)}: {task.get('qupai', '')}")
            piece = self.generate(
                qupai=task["qupai"],
                lyrics=task["lyrics"],
                region=task.get("region"),
                mode=task.get("mode"),
                style_hint=task.get("style_hint"),
                **kwargs,
            )
            results.append(piece)

        return results

    def run_experiment(
        self,
        experiment_name: str,
        tasks: List[Dict],
        prompt_variants: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """
        运行提示工程实验。

        Args:
            experiment_name: 实验名称
            tasks: 任务列表 [{"qupai": ..., "lyrics": ...}, ...]
            prompt_variants: prompt 策略变体列表
                            如 ["zeroshot", "fewshot", "fewshot_tones"]

        Returns:
            {
                "experiment_name": str,
                "results": {variant_name: [GeneratedPiece, ...], ...},
                "summary": {...},
            }
        """
        if prompt_variants is None:
            prompt_variants = list(get_experiment_prompts().keys())

        results = {}
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for variant in prompt_variants:
            logger.info(f"=" * 60)
            logger.info(f"实验变体: {variant}")
            logger.info(f"=" * 60)

            variant_dir = exp_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)

            variant_results = []
            for task in tasks:
                piece = self.generate(
                    qupai=task["qupai"],
                    lyrics=task["lyrics"],
                    prompt_strategy=variant,
                    output_dir=str(variant_dir),
                    **kwargs,
                )
                variant_results.append(piece)

            results[variant] = variant_results

        # 汇总
        summary = self._summarize_experiment(results)
        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"实验完成: {experiment_name}")
        logger.info(f"  摘要: {summary_path}")

        return {
            "experiment_name": experiment_name,
            "results": results,
            "summary": summary,
        }

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _format_output_text(
        self,
        qupai: str,
        lyrics: str,
        gongche_text: str,
        melody_groups: List[LyricNoteGroup],
        qupai_info: Dict,
        examples: List[Dict],
        usage: Dict,
    ) -> str:
        """格式化输出文本文件"""
        lines = []
        lines.append(f"曲牌: {qupai}")
        if qupai_info.get("volume_region"):
            lines.append(f"板块: {qupai_info['volume_region']}")
        if qupai_info.get("volume_mode"):
            lines.append(f"宫调: {qupai_info['volume_mode']}")
        if qupai_info.get("source"):
            lines.append(f"来源风格: {qupai_info['source']}")
        lines.append("")
        lines.append(f"歌词: {lyrics}")

        if melody_groups:
            lines.append(f"旋律: {gongche_text}")
            total_notes = sum(len(g.notes) for g in melody_groups)
            density = total_notes / max(len(melody_groups), 1)
            gongche_seq = []
            for g in melody_groups:
                gongche_seq.extend(n.gongche for n in g.notes)
            lines.append(f"工尺序列: {' '.join(gongche_seq)}")
            lines.append(f"密度: {density:.2f}音/字 ({total_notes}音/{len(melody_groups)}字)")

        lines.append("")
        lines.append(f"参考样例: {[e.get('polyu_id', '') for e in examples]}")
        lines.append(f"Tokens: {usage.get('total_tokens', 0)}")
        lines.append(f"模型: {usage.get('model', 'unknown')}")

        return "\n".join(lines)

    def _summarize_experiment(self, results: Dict[str, List[GeneratedPiece]]) -> Dict:
        """汇总实验结果"""
        summary = {}
        for variant, pieces in results.items():
            successes = [p for p in pieces if p.success]
            summary[variant] = {
                "total": len(pieces),
                "success_count": len(successes),
                "success_rate": len(successes) / max(len(pieces), 1),
                "avg_generation_time": (
                    sum(p.generation_time for p in pieces) / len(pieces)
                    if pieces else 0
                ),
                "avg_tokens": (
                    sum(p.token_usage.get("total_tokens", 0) for p in pieces) / len(pieces)
                    if pieces else 0
                ),
            }
        return summary

    def check_readiness(self) -> Dict:
        """检查生成器就绪状态"""
        status = {
            "dataset_loaded": self.dataset is not None,
            "index_built": self.index is not None,
            "client_ready": False,
            "api_connected": False,
        }

        if self.client:
            status["client_ready"] = True
            try:
                status["api_connected"] = self.client.check_connection()
            except Exception:
                pass

        if self.index:
            status["qupai_count"] = len(self.index)

        return status
