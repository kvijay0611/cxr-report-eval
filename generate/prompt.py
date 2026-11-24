import json
import re
from typing import Optional

from clients.base import Report

SYSTEM_PROMPT = """You are assisting a research evaluation of chest X-ray report drafting.
Return ONLY a JSON object, no other text, with these keys:
  "image_quality": short string,
  "findings": string describing what is visible,
  "impression": string, one or two sentences,
  "needs_more_info": boolean,
  "missing_info": string (what you would need, or "")
Rules:
- Do not invent findings. Only describe what you can actually see.
- If the image is not a chest X-ray, or is too poor to interpret, set
  needs_more_info to true and say so in missing_info instead of guessing.
- This is research output, not medical advice."""


def build_user_prompt(history: Optional[str]) -> str:
    base = "Draft a report for this chest X-ray."
    if history:
        base += f"\nClinical history provided: {history}"
    else:
        base += "\nNo clinical history was provided."
    return base


def parse_report(raw: str) -> Report:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group(0))
            return Report(
                findings=str(d.get("findings", "")).strip(),
                impression=str(d.get("impression", "")).strip(),
                needs_more_info=bool(d.get("needs_more_info", False)),
                missing_info=str(d.get("missing_info", "")).strip(),
                raw=raw,
            )
        except json.JSONDecodeError:
            pass
    # model ignored the format: keep the text so nothing is silently lost
    return Report(findings=raw.strip(), raw=raw)
