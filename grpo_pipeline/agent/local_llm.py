"""
Local LLM wrapper for Qwen2.5-Coder-0.5B-Instruct with LoRA support.

Used by the GRPO trainer to:
  1. Generate Thought→Action→Params completions
  2. Compute per-token log probabilities for GRPO loss (teacher-forcing)
  3. Save / load LoRA checkpoints per training epoch
"""

import os
import sys
from typing import List, Optional, Tuple

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

LORA_CFG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


class LocalLLM:
    """
    Qwen2.5-Coder-0.5B-Instruct with trainable LoRA adapters.

    Key methods:
      generate()          — produce a ReAct completion (no gradient)
      compute_log_probs() — teacher-forcing log probs for GRPO loss (with gradient)
      save_lora()         — persist LoRA weights after each epoch
      load_lora()         — restore weights for eval or continued training
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[LocalLLM] Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        self.model = get_peft_model(base, LORA_CFG)
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

        ref_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        ref_base.eval()
        for p in ref_base.parameters():
            p.requires_grad_(False)
        self.ref_model = ref_base

        print(f"[LocalLLM] Ready. Trainable params above. Device: {self.device}")

    # ------------------------------------------------------------------
    # Generation (no gradient)
    # ------------------------------------------------------------------

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[int]]:
        """
        Generate a ReAct completion.

        Returns:
            (completion_text, completion_token_ids)
            Token IDs are needed by compute_log_probs() for the GRPO loss.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=40,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0][prompt_len:].tolist()
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        del output_ids
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return completion_text, completion_ids

    # ------------------------------------------------------------------
    # Log probability computation (teacher-forcing, keeps gradient)
    # ------------------------------------------------------------------

    def compute_log_probs(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_ids: List[int],
        use_ref: bool = False,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a previously generated completion.

        Uses teacher-forcing so gradients flow back through the policy model.
        When use_ref=True, uses the frozen reference model (no gradients) for
        the KL penalty term in the GRPO loss.

        Returns:
            Tensor of shape (len(completion_ids),) — log p(token_t | context)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(
            self.device
        )
        comp_tensor = torch.tensor(
            completion_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        input_ids = torch.cat([prompt_ids, comp_tensor], dim=1)

        # Truncate to MAX_SEQ_LEN to prevent context explosion across steps
        MAX_SEQ_LEN = 768
        if input_ids.shape[1] > MAX_SEQ_LEN:
            keep_prompt = max(MAX_SEQ_LEN - comp_tensor.shape[1], 1)
            prompt_ids = prompt_ids[:, -keep_prompt:]
            input_ids = torch.cat([prompt_ids, comp_tensor], dim=1)

        model = self.ref_model if use_ref else self.model
        ctx = torch.no_grad() if use_ref else torch.enable_grad()

        with ctx:
            logits = model(input_ids).logits  # (1, seq_len, vocab)

        prompt_len = prompt_ids.shape[1]
        # Positions [prompt_len-1 .. -1] predict tokens [prompt_len .. end]
        completion_logits = logits[0, prompt_len - 1 : -1, :].clone()  # (comp_len, vocab)
        del logits
        log_probs = F.log_softmax(completion_logits, dim=-1)
        del completion_logits
        token_log_probs = log_probs.gather(
            dim=-1, index=comp_tensor.squeeze(0).unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_lora(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[LocalLLM] LoRA checkpoint saved → {path}")

    def load_lora(self, path: str) -> None:
        self.model = PeftModel.from_pretrained(self.model.base_model.model, path)
        self.model.to(self.device)
        print(f"[LocalLLM] LoRA checkpoint loaded ← {path}")
