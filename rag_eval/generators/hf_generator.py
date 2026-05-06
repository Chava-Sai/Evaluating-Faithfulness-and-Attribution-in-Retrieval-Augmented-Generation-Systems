"""HuggingFace-based generator (LLaMA, Mistral)."""

import torch
from typing import List
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, PROMPT_PLAIN


class HFGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 200):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect GPU compute capability to pick the right loading strategy:
        #   sm_80+ (A100, L40S)  → 4-bit NF4 via bitsandbytes (~5 GB)
        #   sm_70  (V100-16GB)   → fp16 directly             (~16 GB, fits exactly)
        #   CPU fallback         → fp32 (slow but safe)
        use_4bit = False
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            use_4bit = major >= 8  # Ampere or newer only
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[HF] GPU: {gpu_name} (sm_{major}x) → "
                  f"{'4-bit NF4' if use_4bit else 'fp16'}")

        print(f"[HF] Loading {model_name} on {device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(CACHE_DIR / "huggingface"),
        )

        if use_4bit:
            # Ampere/Ada (A100, L40S): 4-bit NF4 quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=str(CACHE_DIR / "huggingface"),
            )
        else:
            # Volta (V100) or CPU: load in fp16, let device_map handle placement
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                cache_dir=str(CACHE_DIR / "huggingface"),
            )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, question: str, passages: List[str],
                 prompt_template: str = PROMPT_PLAIN) -> str:
        passage_text = "\n\n".join(
            f"[{i+1}] {p['text']}" for i, p in enumerate(passages)
        )
        prompt = prompt_template.format(
            passages=passage_text,
            question=question,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3000
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded.strip()

    def generate_no_rag(self, question: str) -> str:
        """No-RAG baseline: answer from parametric memory only."""
        prompt = f"Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
