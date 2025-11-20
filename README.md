# 🚀 Language-Neurons-Alignment

Repo of the paper "How does Alignment Enhance LLMs’ Multilingual Capabilities? A Language Neurons Perspective" (AAAI 2026 Oral) 

A complete pipeline for analyzing **language neurons** in multilingual LLMs (Mistral / Llama).
 The toolkit supports:

- Activation extraction (vLLM)
- Prefix vs Full activation decomposition
- Neuron probability estimation
- Entropy-λ neuron scoring
- Language-specific neuron mask construction
- **AutoSweep: search λ such that appear[K] = 0**
- PPL heatmaps under neuron-masked inference

------

## 📦 Installation

```
conda create -n lna python=3.10
conda activate lna
pip install -r requirements.txt
pip install -e .
```

------

## 📁 Project Structure

```
Language-Neurons-Alignment/
│── activations/           # full + prefix activation *.pt
│── activation_masks/      # generated masks
│── ppl_maps/              # PPL heatmaps
│── datasets/              # mgsm.json, etc.
│── language_neurons_alignment/
│   │── activation.py
│   │── analysis.py
│   │── autosweep.py
│   │── cli.py
│   │── config.py
│   │── identify.py
│   │── pipeline.py
│   │── ppl.py
│── requirements.txt
```

------

# 🔧 Usage

All commands use:

```
python -m language_neurons_alignment.cli <command> [...options...]
```

------

# 🟦 1. Collect Activations

Collect **full** and **prefix** activations per language:

```
python -m language_neurons_alignment.cli activation \
  --model-name Mistral \
  --model-path /ABS/PATH/TO/MODEL \
  -l en \
  -d datasets/mgsm.json \
  -s mgsm
```

You must repeat for all languages you want to analyze:

```
en, zh, es, fr, de, ja, ru, bn, th, sw
```

------

# 🟧 2. AutoSweep (Find maximum λ such that appear[K] = 0)

This step performs **binary search** over λ:

```
python -m language_neurons_alignment.cli autosweep \
  --model-name Mistral \
  --model-path /ABS/PATH/TO/MODEL \
  -r 0.01 \
  -s mgsm \
  --lo 0.0 --hi 0.2 --eps 1e-3 \
  -b 0.95
```

Output example:

```
[autosweep] evaluating λ=0.020000
appear = [1024, 380, 51, 4, 0, 0, 0, 0, 0, 0, 0]
best λ = 0.019625
```

------

# 🟩 3. Identify Neurons (full - prefix, then mask generation)

Insert the λ from AutoSweep:

```
python -m language_neurons_alignment.cli identify \
  --model-name Mistral \
  --model-path /ABS/PATH/TO/MODEL \
  -r 0.01 \
  -l <lambda-from-autosweep> \
  -s mgsm \
  -b 0.95
```

Saves:

```
activation_masks/mgsm/0.01-<lambda>/mask.Mistral-<tag>
```

------

# 🟫 4. Compute PPL Maps (with masks applied)

Use the **same** λ:

```
python -m language_neurons_alignment.cli ppl \
  --model-name Mistral \
  --model-path /ABS/PATH/TO/MODEL \
  -r 0.01 \
  -l <lambda-from-autosweep> \
  -s mgsm \
  -d datasets/mgsm.json
```

Outputs:

```
ppl_maps/mgsm/0.01-<lambda>/ppl.Mistral-<tag>.png
```

------

# 🔄 Full Workflow (Recommended)

```
# 1. Activation extraction
for L in en zh es fr de ja ru bn th sw; do
  python -m language_neurons_alignment.cli activation \
    --model-path /ABS/PATH/TO/MODEL \
    -l $L -d datasets/mgsm.json -s mgsm
done

# 2. AutoSweep to find λ*
LAM=$(python - <<EOF
from language_neurons_alignment.autosweep import quick_autosweep
print(quick_autosweep(
  model_name="Mistral",
  model_path="/ABS/PATH/TO/MODEL",
  dataset="mgsm",
  toprate=0.01,
  lo=0.0, hi=0.2, eps=1e-3
))
EOF
)

# 3. Identify by λ*
python -m language_neurons_alignment.cli identify \
  --model-path /ABS/PATH/TO/MODEL \
  -r 0.01 -l $LAM -s mgsm

# 4. PPL masked evaluation
python -m language_neurons_alignment.cli ppl \
  --model-path /ABS/PATH/TO/MODEL \
  -r 0.01 -l $LAM \
  -s mgsm -d datasets/mgsm.json
```

------

# 📜 Citation

```
@article{zhang2025does,
  title={How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective},
  author={Zhang, Shimao and Lai, Zhejian and Liu, Xiang and She, Shuaijie and Liu, Xiao and Gong, Yeyun and Huang, Shujian and Chen, Jiajun},
  journal={arXiv preprint arXiv:2505.21505},
  year={2025}
}
```

