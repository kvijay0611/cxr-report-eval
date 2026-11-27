"""How to talk to a model.

Every model is just a function:   ask(image_paths, history) -> dict
The dict has: text (the report), needs_more_info (True if it declined), missing_info.
Add a new model = add one function and one line in get_model().
"""
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path

PROMPT = """You are assisting a research evaluation of chest X-ray report drafting.
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


class ModelBusy(Exception):
    """Server stayed busy after several retries; the runner skips that study."""


def user_prompt(history):
    text = "Draft one report for this chest X-ray study (all attached images belong to the same exam)."
    return text + (f"\nClinical history provided: {history}" if history else "\nNo clinical history was provided.")


def parse_report(raw):
    """Model output -> dict. If the model ignored the JSON format, keep its raw text."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            text = f"{d.get('findings', '')} {d.get('impression', '')}".strip()
            return {"text": text, "needs_more_info": bool(d.get("needs_more_info", False)),
                    "missing_info": str(d.get("missing_info", ""))}
        except json.JSONDecodeError:
            pass
    return {"text": raw.strip(), "needs_more_info": False, "missing_info": ""}


# ---- model 1: control that ignores the image ---------------------------------
def constant_model(image_paths, history=None):
    return {"text": "The heart size and mediastinum are within normal limits. The lungs are clear. "
                    "No pleural effusion or pneumothorax. No acute cardiopulmonary abnormality.",
            "needs_more_info": False, "missing_info": ""}


# ---- model 2: any OpenAI-style API (Gemini, Groq, vLLM, ...) -----------------
def openai_model(model, base_url=None):
    from openai import OpenAI
    client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), max_retries=0, timeout=120)

    def ask(image_paths, history=None):
        content = [{"type": "text", "text": user_prompt(history)}]
        for p in image_paths or []:
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        last = None
        for attempt in range(6):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=700,
                    messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": content}])
                return parse_report(r.choices[0].message.content or "")
            except Exception as e:
                busy = (getattr(e, "status_code", None) in (429, 500, 502, 503, 504)
                        or type(e).__name__ in ("APIConnectionError", "APITimeoutError"))
                if not busy:
                    raise                       # wrong model name, bad key, etc: stop and show it
                if "per day" in str(e).lower() or "perday" in str(e).lower():
                    raise SystemExit("Daily free quota used up. Wait for the reset, switch model "
                                     "or provider, then rerun (finished studies are kept).")
                last = e
                time.sleep(min(60, 3 * 2 ** attempt))
        raise ModelBusy(f"{type(last).__name__}: {str(last)[:200]}")
    return ask


# ---- model 3: a Hugging Face model run on your own GPU (e.g. Kaggle) ---------
def hf_model(model_id="google/medgemma-4b-it", dtype="bfloat16", max_images=1):
    """Needs HF_TOKEN for gated models. Written without GPU access: expect small fixes."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch_dtype = getattr(torch, dtype)
    token = os.getenv("HF_TOKEN")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto", token=token)
    processor = AutoProcessor.from_pretrained(model_id, token=token)

    def ask(image_paths, history=None):
        images = [Image.open(p).convert("RGB") for p in (image_paths or [])][:max_images]
        content = [{"type": "text", "text": user_prompt(history)}]
        content += [{"type": "image", "image": im} for im in images]
        messages = [{"role": "system", "content": [{"type": "text", "text": PROMPT}]},
                    {"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(model.device, dtype=torch_dtype)
        n_in = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        return parse_report(processor.decode(out[0][n_in:], skip_special_tokens=True))
    return ask


def get_model(kind, name="", base_url=None, dtype="bfloat16", max_images=1):
    if kind == "constant":
        return constant_model
    if kind == "openai":
        return openai_model(name, base_url)
    if kind == "hf":
        return hf_model(name or "google/medgemma-4b-it", dtype, max_images)
    raise ValueError(kind)
