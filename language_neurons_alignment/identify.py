import os
import numpy as np
import torch

from .config import (
    set_model_name,
    get_suffix_from_path,
    LANG_MAPPING,
)
from .analysis import Neuron_Prob, Neuron_Mask

class ActivationAnalyzer:
    def __init__(self, model_name: str, model_path: str, langs):
        self.model_name = model_name
        assert isinstance(model_path, str), f"model_path must be str, got {type(model_path)}"
        self.model_path = model_path
        self.langs = list(langs)

    def run(
        self,
        dataset: str,
        top_rate: float,
        xlambda: float,
        activation_bar_ratio: float = 0.95,
    ) -> Neuron_Mask:
        prob = Neuron_Prob(model_name=self.model_name, model_path=self.model_path, langs=self.langs, dataset=dataset)
        num_layers, inter_size, _ = prob.activation_probs.shape

        entropy = torch.zeros((num_layers, inter_size))
        norm_max = sum(1 / k for k in range(1, len(self.langs) + 1)) / len(self.langs)

        for l in range(num_layers):
            for n in range(inter_size):
                probs = prob.wrap_neuron(l, n, "normed")
                e = Neuron_Prob.calc_entropy(probs) / np.log(len(self.langs))
                max_act = prob.wrap_neuron(l, n, "activation").max().item()
                entropy[l, n] = e - max_act * float(xlambda) / norm_max

        k = round(entropy.numel() * float(top_rate))
        _, flat_idx = entropy.flatten().topk(k, largest=False)
        rows, cols = flat_idx // entropy.size(1), flat_idx % entropy.size(1)

        thr = prob.activation_probs.flatten().kthvalue(
            round(prob.activation_probs.numel() * activation_bar_ratio)
        ).values.item()

        mask_data = [[] for _ in self.langs]
        for i in range(len(rows)):
            l = rows[i].item()
            n = cols[i].item()
            for lid in range(len(self.langs)):
                if prob.activation_probs[l, n, lid] > thr:
                    while len(mask_data[lid]) <= l:
                        mask_data[lid].append([])
                    mask_data[lid][l].append(n)

        for lid in range(len(self.langs)):
            for l in range(num_layers):
                if l < len(mask_data[lid]):
                    mask_data[lid][l] = torch.tensor(sorted(set(mask_data[lid][l]))).long()
                else:
                    mask_data[lid].append(torch.tensor([], dtype=torch.long))

        return Neuron_Mask(mask_data, self.langs)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate language-specific neuron masks (identify)")
    ap.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("-r", "--top-rate", type=float, default=0.10)
    ap.add_argument("-l", "--xlambda", type=float, default=0.00)
    ap.add_argument("-s", "--dataset", type=str, default="mgsm")
    ap.add_argument("-b", "--bar", type=float, default=0.95, help="activation_bar_ratio")
    args = ap.parse_args()

    langs = list(LANG_MAPPING.keys())
    analyzer = ActivationAnalyzer(args.model_name, args.model_path, langs)
    mask = analyzer.run(args.dataset, args.top_rate, args.xlambda, args.bar)

    folder = f"{args.top_rate}-{args.xlambda}"
    os.makedirs(f"activation_masks/{args.dataset}/{folder}", exist_ok=True)
    tag = get_suffix_from_path(args.model_path)
    out = f"activation_masks/{args.dataset}/{folder}/mask.{args.model_name}{tag}"
    mask.save_file(out)
    print("Mask saved to:", out)

if __name__ == "__main__":
    main()
