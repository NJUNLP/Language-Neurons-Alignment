import os
import json
import sys
from types import MethodType
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .config import resolve_model_path, get_suffix_from_path, LANG_MAPPING


class ActivationProfiler:
    def __init__(self, model_name: str, model_path: str, lang_code: str):
        self.model_name = model_name
        self.model_path = resolve_model_path(model_path)
        self.lang_code = lang_code
        self.lang_name = LANG_MAPPING.get(lang_code, lang_code)  # 兼容直接传英文名
        self.model = self._init_model()

        cfg = self.model.llm_engine.model_config.hf_config
        self.num_layers = int(getattr(cfg, "num_hidden_layers"))
        self.intermediate_size = int(getattr(cfg, "intermediate_size"))
        self.over_zero = torch.zeros(self.num_layers, self.intermediate_size,
                                     dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_activation_hooks()

    def _init_model(self) -> LLM:
        tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"[Activation] Loading model from: {self.model_path}")
        return LLM(
            model=self.model_path,
            tensor_parallel_size=tp,
            enforce_eager=True,
            max_model_len=4096,
        )

    def _hf_layers(self):
        hf = self.model.llm_engine.model_executor.driver_worker.model_runner.model.model
        return hf.layers

    def _get_mlp_layer(self, layer_idx: int):
        return self._hf_layers()[layer_idx].mlp

    def _create_forward_method(self, layer_idx: int):
        profiler = self

        def mistral_forward(self, x):
            out = self.gate_up_proj(x)
            gate_up = out[0] if isinstance(out, tuple) else out  # [..., 2D]
            d = gate_up.shape[-1] // 2

            gate = gate_up[..., :d]
            up = gate_up[..., d:]

            gate = F.silu(gate)

            activation = gate.float()              # [..., D]
            pos = activation > 0                  # bool [..., D]

            pos_flat = pos.reshape(-1, pos.shape[-1])   # [N, D]
            cnt = pos_flat.sum(dim=0)                   # [D]

            profiler.over_zero[layer_idx, :] += cnt.to(profiler.over_zero.dtype)

            x = gate * up
            out = self.down_proj(x)
            return out[0] if isinstance(out, tuple) else out

        return mistral_forward

    def _prepare_activation_hooks(self):
        for li in range(self.num_layers):
            mlp = self._get_mlp_layer(li)
            mlp.forward = MethodType(self._create_forward_method(li), mlp)

    def _iter_prompts(self, data_path: str, prefix_only: bool) -> List[str]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if item.get("lang") == self.lang_name:
                if prefix_only:
                    instr = item.get("instruction", "")
                    for _ in item.get("answers", []):
                        prompts.append(instr)
                else:
                    instr = item.get("instruction", "")
                    for ans in item.get("answers", []):
                        gen = ans.get("generated", "")
                        prompts.append(f"{instr}{gen}")
        return [p for p in prompts if p]

    def _encode_lengths(self, toks, tokenizer) -> int:
        return sum(len(tokenizer.encode(p)) for p in toks)

    def _run_and_save(self, prompts: List[str], dataset_name: str, lang_tag: str, is_prefix: bool):
        os.makedirs(os.path.join("activations", dataset_name), exist_ok=True)

        tokenizer = self.model.get_tokenizer()
        total_tokens = self._encode_lengths(prompts, tokenizer)

        token_ids = [tokenizer.encode(p) for p in prompts]
        sampling = SamplingParams(temperature=0, max_tokens=1)
        _ = self.model.generate(prompt_token_ids=token_ids, sampling_params=sampling)

        tag = get_suffix_from_path(self.model_path)
        prefix = "prefix." if is_prefix else ""
        out_path = f"activations/{dataset_name}/{prefix}activation.{lang_tag}.{self.model_name}{tag}.pt"

        payload = {
            "total_tokens": int(total_tokens),
            "over_zero": self.over_zero.detach().cpu(),  # [L, D] int64
        }
        torch.save(payload, out_path)
        print(f"[Activation] saved -> {out_path}")
        # print(f"[Activation] total tokens: {total_tokens}, max over_zero: {int(self.over_zero.max())}")

    def process_dataset(self, data_path: str, dataset_name: str):
        prompts = self._iter_prompts(data_path, prefix_only=False)
        if not prompts:
            raise ValueError(f"No prompts found for lang={self.lang_name}")
        self._run_and_save(prompts, dataset_name, self.lang_code, is_prefix=False)

    def get_prefix_dataset(self, data_path: str, dataset_name: str):
        self.over_zero.zero_()
        prompts = self._iter_prompts(data_path, prefix_only=True)
        if not prompts:
            raise ValueError(f"No prefix prompts found for lang={self.lang_name}")
        self._run_and_save(prompts, dataset_name, self.lang_code, is_prefix=True)


def collect_activation(
    model_name: str,
    model_path: str,
    lang: str,
    dataset_path: str,
    dataset: str,
):
    profiler = ActivationProfiler(
        model_name=model_name,
        model_path=model_path,
        lang_code=lang,
    )
    # 全文：instruction + generated
    profiler.process_dataset(dataset_path, dataset)
    # 前缀：只 instruction（重置 over_zero）
    profiler.get_prefix_dataset(dataset_path, dataset)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Language Model Activation Profiler (vLLM hook, original logic)")
    parser.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    parser.add_argument("--model-path", required=True, help="HF/vLLM model directory or HF id")
    parser.add_argument("-l", "--lang", type=str, default="en", help="Language code in LANG_MAPPING (e.g., en)")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to input dataset JSON")
    parser.add_argument("-s", "--dataset", type=str, default="mgsm", help="Dataset tag for output dir")
    args = parser.parse_args()

    try:
        collect_activation(
            model_name=args.model_name,
            model_path=args.model_path,
            lang=args.lang,
            dataset_path=args.data,
            dataset=args.dataset,
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()