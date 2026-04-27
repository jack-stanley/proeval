# ProEval

ProEval: Proactive failure discovery and efficient performance estimation for GenAI evaluation.

1. 💰 **Cut GenAI eval costs 8–100×** — achieve ±1% accuracy with a fraction of the samples
2. 🔍 **Discover failure cases** — proactively surface diverse bugs under strict evaluation budgets
3. 🧠 **Transfer learning over benchmarks** — pre-trained Gaussian Process surrogates generalize to new models instantly
4. 🧩 **Easy Integration** - Easily to integrate into the GenAI evaluation systems with multiple modalities
5. ✅ **Validated on reasoning, safety & classification** — GSM8K, MMLU, StrategyQA, Jigsaw, and more

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from proeval import BQPriorSampler, LLMPredictor, DATASET_CONFIGS

# Estimate a model's error rate with ~1% of the data
sampler = BQPriorSampler(noise_variance=0.3)
result = sampler.sample(predictions="svamp", target_model="gemini25_flash", budget=50)
print(f"Estimated error rate: {result.estimates[-1]:.4f}")
print(f"MAE: {result.mae(true_mean):.4f}")

# Evaluate an LLM on a supported benchmark
predictor = LLMPredictor(model="google/gemma-3-4b-it")
response, prediction, score = predictor.evaluate(
    "Is the sky blue?", True, DATASET_CONFIGS["strategyqa"]
)
```

## Experiments

Here is an example of how to run the experiments:

```shell
# BQ sampling with learned prior (Case 1)
python -m experiment.exp_sampling_w_prior_case_1 --dataset svamp --n-runs 5
```

## License

```
Copyright 2026 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license. You
may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
```
