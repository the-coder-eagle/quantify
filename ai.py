"""AI client for DeepSeek API (OpenAI-compatible)."""
import json, time
from loguru import logger

class AIClient:
    def __init__(self, api_key="", base_url="https://api.deepseek.com", model="deepseek-v4-flash", max_retries=3):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model; self.max_retries = max_retries

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=1536):
        msgs = [{"role":"system","content":system_prompt},{"role":"user","content":user_message}]
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                            temperature=temperature, max_tokens=max_tokens)
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1: time.sleep(2**attempt)
                else: logger.error(f"AI failed: {e}"); return f"[AI Error]"
        return ""

    def ask_json(self, system_prompt, user_message, temperature=0.3):
        reply = self.chat(system_prompt + "\nReturn ONLY valid JSON.", user_message, temperature)
        try:
            if "```json" in reply: reply = reply.split("```json")[1].split("```")[0]
            elif "```" in reply: reply = reply.split("```")[1].split("```")[0]
            return json.loads(reply.strip())
        except Exception:
            logger.debug(f"JSON parse failed on AI reply, returning raw")
            return {"raw": reply}
