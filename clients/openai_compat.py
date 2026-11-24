import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from generate.prompt import SYSTEM_PROMPT, build_user_prompt, parse_report

from .base import Report, ReportClient


class OpenAICompatClient(ReportClient):
    """Works with any server that speaks the OpenAI chat API with images:
    hosted APIs, vLLM, SGLang, etc. Point base_url at the server."""

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.0):
        self.name = model
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        )

    def generate(self, image_path: Optional[Path], history: Optional[str] = None) -> Report:
        content = [{"type": "text", "text": build_user_prompt(history)}]
        if image_path is not None:
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=700,
        )
        return parse_report(resp.choices[0].message.content or "")
