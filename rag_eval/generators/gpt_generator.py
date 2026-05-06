"""GPT-3.5-turbo API generator (baseline)."""

import os
from typing import List
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROMPT_PLAIN


class GPTGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 200):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def generate(self, question: str, passages: List[str],
                 prompt_template: str = PROMPT_PLAIN) -> str:
        passage_text = "\n\n".join(
            f"[{i+1}] {p['text']}" for i, p in enumerate(passages)
        )
        prompt = prompt_template.format(passages=passage_text, question=question)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def generate_no_rag(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Answer concisely: {question}"}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
