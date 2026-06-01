"""
项目配置文件，管理路径、API密钥、模型参数等。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据路径
DATA_DIR = ROOT_DIR
SONGS_CSV = DATA_DIR / "FINAL_SONGS.csv"
NOTES_CSV = DATA_DIR / "FINAL_NOTES.csv"
BEATS_CSV = DATA_DIR / "FINAL_BEATS.csv"
MUSICXML_DIR = DATA_DIR / "musicxml"

# 输出路径
OUTPUT_DIR = ROOT_DIR / "experiments" / "outputs"
PROMPTS_DIR = ROOT_DIR / "experiments" / "prompts"


@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"  # 或 deepseek-reasoner
    temperature: float = 0.8
    max_tokens: int = 2048
    top_p: float = 0.95
    max_retries: int = 3
    retry_delay: float = 2.0  # 秒


@dataclass
class GeneratorConfig:
    """生成器配置"""
    n_fewshot_examples: int = 3
    output_musicxml: bool = True
    output_meta_json: bool = True
    default_region: Optional[str] = None  # None = 从曲牌推断
    default_mode: Optional[str] = None


@dataclass
class Config:
    """总配置"""
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    prompts_dir: Path = PROMPTS_DIR

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """从 .env 文件和环境变量加载配置"""
        if env_path:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        else:
            from dotenv import load_dotenv
            load_dotenv(ROOT_DIR / ".env", override=False)

        return cls(
            deepseek=DeepSeekConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.8")),
            ),
            output_dir=Path(os.getenv("OUTPUT_DIR", str(OUTPUT_DIR))),
        )


# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
