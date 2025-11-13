import os
import torch
from .config import LANG_MAPPING, get_suffix_from_path, resolve_model_path
from .identify import ActivationAnalyzer
from .analysis import Neuron_Mask

def run_identify_and_count(
    model_name: str,
    model_path: str,
    dataset: str,
    toprate: float,
    lam: float,
    activation_bar_ratio: float = 0.95,
):
    model_path = resolve_model_path(model_path)
    langs = list(LANG_MAPPING.keys())

    analyzer = ActivationAnalyzer(model_name=model_name, model_path=model_path, langs=langs)
    mask = analyzer.run(dataset, toprate, lam, activation_bar_ratio)

    folder = f"{toprate}-{lam}"
    os.makedirs(f"activation_masks/{dataset}/{folder}", exist_ok=True)
    tag = get_suffix_from_path(model_path)
    mask_path = f"activation_masks/{dataset}/{folder}/mask.{model_name}{tag}"
    mask.save_file(mask_path)

    appear = mask.appear_distribution()
    return mask_path, appear
