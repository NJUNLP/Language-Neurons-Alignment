from __future__ import annotations

import os
import math
import json
from typing import List, Tuple, Dict, Sequence

import torch
import numpy as np

from .config import LANG_MAPPING, get_suffix_from_path


class Neuron_Prob:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        dataset: str,
        langs: Sequence[str] | None = None,
        use_delta: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.dataset = dataset
        self.langs = list(langs) if langs is not None else list(LANG_MAPPING.keys())
        self.use_delta = use_delta
        self.device = torch.device(device)

        self.activation_probs: torch.Tensor  # [L, H, K]
        self.num_layers: int = 0
        self.intermediate_size: int = 0

        self._load_and_build()

    @staticmethod
    def calc_entropy(probs_1d: torch.Tensor, eps: float = 1e-12) -> float:
        p = probs_1d.clamp_min(eps)
        p = p / p.sum()  # 再保险
        return float(-(p * (p + 1e-30).log()).sum().item())

    def wrap_neuron(self, l: int, n: int, mode: str = "normed") -> torch.Tensor:
        vec = self.activation_probs[l, n, :]  # (K,)
        if mode == "activation":
            return vec
        elif mode == "normed":
            s = vec.sum()
            if s <= 0:
                return torch.full_like(vec, 1.0 / vec.numel())
            return vec / s
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _load_one_stats(self, lang_code: str, tag: str) -> Tuple[int, torch.Tensor]:
        base = f"activations/{self.dataset}"
        path = os.path.join(base, f"activation.{lang_code}.{self.model_name}{tag}.pt")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        total_tokens = int(obj["total_tokens"])
        over_zero = obj["over_zero"].to(torch.long)
        return total_tokens, over_zero

    def _load_one_prefix(self, lang_code: str, tag: str) -> Tuple[int, torch.Tensor]:
        base = f"activations/{self.dataset}"
        path = os.path.join(base, f"prefix.activation.{lang_code}.{self.model_name}{tag}.pt")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        total_tokens = int(obj["total_tokens"])
        over_zero = obj["over_zero"].to(torch.long)
        return total_tokens, over_zero

    def _load_and_build(self) -> None:
        tag = get_suffix_from_path(self.model_path)

        if len(self.langs) == 0:
            raise ValueError("langs is empty.")

        full_tok_0, full_oz_0 = self._load_one_stats(self.langs[0], tag)
        pre_tok_0, pre_oz_0 = self._load_one_prefix(self.langs[0], tag)

        L, H = full_oz_0.shape
        self.num_layers, self.intermediate_size = L, H

        K = len(self.langs)
        act = torch.zeros((L, H, K), dtype=torch.float32)

        d_tok0 = max(full_tok_0 - pre_tok_0, 1)
        d_oz0 = (full_oz_0 - pre_oz_0).clamp_min(0)
        act[:, :, 0] = d_oz0.to(torch.float32) / float(d_tok0)

        for k, lang in enumerate(self.langs[1:], start=1):
            full_tok, full_oz = self._load_one_stats(lang, tag)
            pre_tok, pre_oz = self._load_one_prefix(lang, tag)

            if full_oz.shape != (L, H) or pre_oz.shape != (L, H):
                raise RuntimeError(
                    f"Shape mismatch for lang={lang}. "
                    f"Expected {(L, H)}, got full={tuple(full_oz.shape)}, prefix={tuple(pre_oz.shape)}"
                )

            d_tok = max(full_tok - pre_tok, 1)
            d_oz = (full_oz - pre_oz).clamp_min(0)
            act[:, :, k] = d_oz.to(torch.float32) / float(d_tok)

        self.activation_probs = act.to(self.device)


class Neuron_Mask:
    def __init__(self, mask_data: List[List[torch.Tensor]], langs: Sequence[str]) -> None:
        self.mask_data = mask_data
        self.langs = list(langs)

    def save_file(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(self.mask_data, out_path)

    def size(self, mode: str = "total"):
        """
        - "total": return all languages
        - "per_lang": return {lang_code: count}
        """
        if mode == "total":
            s = 0
            for lang_layers in self.mask_data:
                for layer_tensor in lang_layers:
                    s += int(layer_tensor.numel())
            return s

        elif mode == "per_lang":
            stats = {}
            for lid, code in enumerate(self.langs):
                cnt = 0
                for t in self.mask_data[lid]:
                    cnt += int(t.numel())
                stats[code] = cnt
            return stats

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def appear_distribution(self) -> List[int]:
        """
        count appear[0..K]
        """
        K = len(self.langs)
        if not self.mask_data:
            return [0] * (K + 1)

        L = max(len(lang_layers) for lang_layers in self.mask_data)

        appear = [0] * (K + 1)

        for l in range(L):
            counter: Dict[int, int] = {}
            for lid in range(K):
                if l < len(self.mask_data[lid]):
                    layer_tensor = self.mask_data[lid][l]
                    if layer_tensor.numel() == 0:
                        continue
                    for o in layer_tensor.tolist():
                        counter[o] = counter.get(o, 0) + 1

            for c in counter.values():
                if 0 <= c <= K:
                    appear[c] += 1

        return appear
