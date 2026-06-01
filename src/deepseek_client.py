"""
DeepSeek API 客户端模块。

使用 OpenAI 兼容接口调用 DeepSeek Chat API。
"""

import time
import logging
from typing import Optional, List, Dict, Tuple
from openai import OpenAI

from .config import DeepSeekConfig

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    DeepSeek API 客户端封装。

    Usage:
        config = DeepSeekConfig(api_key="sk-xxx")
        client = DeepSeekClient(config)
        response = client.chat(system_prompt, user_prompt)
    """

    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "DeepSeek API key 未设置。"
                    "请设置环境变量 DEEPSEEK_API_KEY 或在 config 中提供。"
                )
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """
        发送 Chat Completion 请求。

        Args:
            system_prompt: 系统提示
            user_prompt: 用户消息
            temperature: 温度（默认用配置值）
            max_tokens: 最大输出 token
            model: 模型名（默认用配置值）
            top_p: top_p 采样参数

        Returns:
            (response_text, usage_info)

        Raises:
            RuntimeError: API 调用失败
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        mdl = model or self.config.model
        tp = top_p or self.config.top_p

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=mdl,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=tp,
                )

                text = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "model": response.model,
                }

                logger.info(f"API 调用成功: {usage['total_tokens']} tokens")
                return text, usage

            except Exception as e:
                last_error = e
                logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"  等待 {delay}s 后重试...")
                    time.sleep(delay)

        raise RuntimeError(
            f"DeepSeek API 调用失败（已重试{self.config.max_retries}次）: {last_error}"
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        **kwargs,
    ) -> str:
        """
        简化的生成接口 — 仅返回文本。

        Returns:
            生成的文本
        """
        text, _ = self.chat(system_prompt, user_prompt, **kwargs)
        return text

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],  # [(system_prompt, user_prompt), ...]
        **kwargs,
    ) -> List[Tuple[str, dict]]:
        """
        批量生成（顺序执行，无并发）。

        Args:
            prompts: [(system_prompt, user_prompt), ...]
            **kwargs: 传递给 chat() 的额外参数

        Returns:
            [(response_text, usage_info), ...]
        """
        results = []
        for i, (sys_p, usr_p) in enumerate(prompts):
            logger.info(f"批量生成 {i + 1}/{len(prompts)}")
            try:
                result = self.chat(sys_p, usr_p, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"  第 {i + 1} 项失败: {e}")
                results.append(("", {"error": str(e)}))
        return results

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量（粗略估计）。

        中文: ~1.5 字符/token
        英文: ~4 字符/token
        """
        # 粗略估算
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def check_connection(self) -> bool:
        """测试 API 连接"""
        try:
            self.chat(
                system_prompt="你是一个助手。",
                user_prompt="请回复'连接成功'四个字。",
                max_tokens=10,
            )
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
