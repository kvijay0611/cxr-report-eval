import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import List, Optional

import openai
from openai import OpenAI

from generate.prompt import SYSTEM_PROMPT, build_user_prompt, parse_report

from .base import Report, ReportClient, TransientModelError


# busy server (503), rate limit (429), network trouble: worth retrying
TRANSIENT_ERRORS = (openai.RateLimitError, openai.InternalServerError,
                    openai.APIConnectionError, openai.APITimeoutError)


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
            max_retries=0,          # we do our own retrying below
            timeout=120,
        )

    def generate(self, image_paths: Optional[List[Path]], history: Optional[str] = None) -> Report:
        content = [{"type": "text", "text": build_user_prompt(history)}]
        for image_path in image_paths or []:
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        last = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=700,
                )
                return parse_report(resp.choices[0].message.content or "")
            except TRANSIENT_ERRORS as e:      # busy server (503), rate limit (429), network
                last = e
                time.sleep(min(60, 3 * 2 ** attempt))
        raise TransientModelError(f"{type(last).__name__}: {str(last)[:200]}") from last