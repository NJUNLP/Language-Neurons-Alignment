import os
import torch
import numpy as np
import json
from types import MethodType
import torch.nn.functional as F
from vllm import LLM, SamplingParams

from .config import LANG_MAPPING, get_suffix_from_path, resolve_model_path

class PerplexityAnalyzer:
    def __init__(self, model_name: str, model_path: str, lang_keys):
        self.model_name = model_name
        self.model_path = resolve_model_path(model_path)
        self.lang = list(lang_keys)
        self.lang_mapping = {k: LANG_MAPPING[k] for k in self.lang}
        self.model = self._load_model()
        self.activation_masks = None

    def _load_model(self) -> LLM:
        return LLM(
            model=self.model_path,
            dtype=torch.bfloat16,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85
        )

    def _get_mlp_layer(self, layer_idx):
        return self.model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx].mlp

    def load_activation_masks(self, mask_path):
        if not mask_path:
            return
        self.activation_masks = torch.load(mask_path, weights_only=True)

    def apply_activation_mask(self, lang_x):
        if not lang_x or not self.activation_masks:
            print("No activation mask applied")
            return
        lang_idx = self.lang.index(lang_x)
        activation_mask = self.activation_masks[lang_idx]
        for layer_idx, layer_mask in enumerate(activation_mask):
            mlp_layer = self._get_mlp_layer(layer_idx)
            mlp_layer.forward = MethodType(
                self._create_masked_forward(layer_mask),
                mlp_layer
            )

    def _create_masked_forward(self, mask):
        device_mask = torch.tensor(mask).to('cuda').to(torch.int64)
        def masked_forward(mlp_self, x: torch.Tensor) -> torch.Tensor:
            gate_up, _ = mlp_self.gate_up_proj(x)
            d = gate_up.shape[-1] // 2
            activation = F.silu(gate_up[..., :d])
            activation.index_fill_(-1, device_mask, 0)
            x = activation * gate_up[..., d:]
            x, _ = mlp_self.down_proj(x)
            return x
        return masked_forward

    def _load_dataset(self, path, lang_y):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        prompts = []
        for item in dataset:
            if item.get("lang") == self.lang_mapping.get(lang_y):
                prompts.append(item.get("instruction", ""))
        return prompts

    def _calculate_batch_ppl(self, input_ids):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=1
        )
        outputs = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params
        )
        return [self._extract_ppl(output) for output in outputs]

    def _extract_ppl(self, output) -> float:
        logprobs = []
        for step_logprobs in output.prompt_logprobs[1:]:
            if step_logprobs is None:
                continue
            for _, logprob_obj in step_logprobs.items():
                logprobs.append(logprob_obj.logprob)
        if not logprobs:
            return float('inf')
        nll = -np.mean(logprobs)
        return nll

    def calculate_perplexity(self, lang_y, datapath):
        prompts = self._load_dataset(datapath, lang_y)
        tokenizer = self.model.get_tokenizer()
        max_length = self.model.llm_engine.model_config.max_model_len
        batch_size = 1
        total_ppl = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            input_ids = [
                tokenizer.encode(p, max_length=max_length, truncation=True)
                for p in batch_prompts
            ]
            batch_ppl = self._calculate_batch_ppl(input_ids)
            total_ppl.extend(batch_ppl)
        return np.exp(np.mean(total_ppl))

def run_grid_ppl(model_name: str, model_path: str, dataset: str, toprate: str, xlambda: str, data_path: str):
    langs = list(LANG_MAPPING.keys())
    tag = get_suffix_from_path(model_path)

    folder = f"{toprate}-{xlambda}"
    mask_path = f"activation_masks/{dataset}/{folder}/mask.{model_name}{tag}"
    print(f"[PPL] load mask from: {mask_path} (if exists)")

    analyzer = PerplexityAnalyzer(model_name, model_path, langs)
    if os.path.exists(mask_path):
        analyzer.load_activation_masks(mask_path)

    os.makedirs(f"ppl_maps/{dataset}/{folder}", exist_ok=True)
    result_path = f"ppl_maps/{dataset}/{folder}/ppl.{model_name}{tag}"
    ppl_data = torch.load(result_path, weights_only=False) if os.path.exists(result_path) else {}

    for deactivate_lang in [None] + langs:
        ppl_data.setdefault(deactivate_lang, {})
        for target_lang in langs:
            if target_lang not in ppl_data[deactivate_lang]:
                analyzer.apply_activation_mask(deactivate_lang)
                ppl = analyzer.calculate_perplexity(target_lang, data_path)
                ppl_data[deactivate_lang][target_lang] = ppl
                torch.save(ppl_data, result_path)
                print(f"Saved {deactivate_lang}->{target_lang}: {ppl:.3f}")

    print(result_path)
    print(ppl_data)
