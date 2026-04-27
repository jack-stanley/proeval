# ProEval Dataset

This dataset contains the evaluation results used for validating ProEval, a sample-efficient framework combining transfer learning and Bayesian modeling to accurately estimate performance and proactively identify failure cases.

More details of the dataset can be found in the following paper:


> Yizheng Huang, Wenjun Zeng, Aditi Kumaresan and Zi Wang. ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation. Technical report, Google DeepMind, 2026.


## Directory Structure

```
data/
├── <benchmark>_predictions.csv                         # Model predictions per benchmark
├── <benchmark>_embeddings_text_embedding_3_large.npy   # Question embeddings (OpenAI text-embedding-3-large)
└── README.md                                           # This file
```

## Benchmarks

| Dataset      | Domain                    | # Rows | # Models | Label Type |
| ------------ | ------------------------- | ------ | -------- | ---------- |
| `gsm8k`      | Math reasoning            | ~1,319 | 16       | Binary     |
| `svamp`      | Math word problems        | 700    | 16       | Binary     |
| `mmlu`       | Multi-task knowledge      | ~1,534 | 16       | Binary     |
| `strategyqa` | Yes/no reasoning          | ~1,603 | 16       | Binary     |
| `gqa`        | Visual question answering | ~2,000 | 14       | Binary     |
| `jigsaw`     | Toxicity detection        | ~1,500 | 16       | Binary     |
| `dices`      | Safety rating             | ~1,500 | 16       | Continuous |
| `dices_t2i`  | Text-to-image safety      | ~1,500 | 14       | Continuous |

> [!NOTE]
> Row counts refer to the number of unique questions (after header). The actual CSV
> line counts are larger because multi-line content (e.g., JSON in `raw_response`)
> spans several lines within a single record.

---

## Predictions CSV Format

All `*_predictions.csv` files share the same columnar schema. Each file is a
standard CSV with one row per evaluation example. Columns are organized into
**shared columns** (the first 3) followed by **per-model column groups** that
repeat for every evaluated model.

### Shared Columns

| Column         | Type  | Description                                                                                                                                                        |
| -------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `index`        | `int` | Zero-based row index — unique identifier for each evaluation example within the benchmark.                                                                         |
| `question`     | `str` | The input prompt or question sent to each model. Content varies by benchmark (see [Question Format by Benchmark](#question-format-by-benchmark) below).            |
| `ground_truth` | `str` | The reference answer or ground-truth label for the example. Interpretation depends on the benchmark (see [Ground Truth Semantics](#ground-truth-semantics) below). |

### Per-Model Column Groups

For every evaluated model `<model>`, the following three columns appear:

| Column                 | Type    | Description                                                                                                                                                                                                                                                                                           |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prediction_<model>`   | `str`   | The model's **extracted answer** — a short, parsed output (e.g., a number, a letter choice, `"yes"`/`"no"`, or a rating). This is the value actually compared against the ground truth.                                                                                                               |
| `label_<model>`        | `float` | The **error score** for this prediction. Measures how far the prediction deviates from the ground truth. `0.0` = fully correct, `1.0` = fully incorrect. For binary benchmarks this is `{0.0, 1.0}`; for continuous benchmarks (DICES, DICES-T2I) this takes values in `{0.0, 0.25, 0.5, 0.75, 1.0}`. |
| `raw_response_<model>` | `str`   | The model's **full JSON response** before answer extraction. Typically contains `"reasoning"` and `"answer"` (or `"rating"`) fields. This preserves the complete chain-of-thought for auditing and debugging.                                                                                         |

### Evaluated Models

Up to **16 models** are evaluated per benchmark. The full model roster is:

| Column Suffix     | Model             |
| ----------------- | ----------------- |
| `gpt35_turbo`     | GPT-3.5 Turbo     |
| `gpt_4o`          | GPT-4o            |
| `gpt5`            | GPT-5             |
| `gpt5_1`          | GPT-5.1           |
| `gpt5_2`          | GPT-5.2           |
| `claude35_haiku`  | Claude 3.5 Haiku  |
| `claude37_sonnet` | Claude 3.7 Sonnet |
| `claude45_sonnet` | Claude 4.5 Sonnet |
| `claude45_opus`   | Claude 4.5 Opus   |
| `gemini25_flash`  | Gemini 2.5 Flash  |
| `gemini25_pro`    | Gemini 2.5 Pro    |
| `gemini3_flash`   | Gemini 3 Flash    |
| `gemini3_pro`     | Gemini 3 Pro      |
| `gemma3_12b`      | Gemma 3 12B       |
| `gemma3_27b`      | Gemma 3 27B       |
| `qwen3_32b`       | Qwen 3 32B        |

> [!NOTE]
> `gqa` and `dices_t2i` (DIVE) include 14 models (missing `gpt35_turbo` and `qwen3_32b`).
> All other benchmarks include all 16 models.

---

## Question Format by Benchmark

The `question` column encodes different information depending on the benchmark:

| Benchmark    | Format                                     | Example                                                                    |
| ------------ | ------------------------------------------ | -------------------------------------------------------------------------- |
| `gsm8k`      | Plain text math word problem               | `"Janet's ducks lay 16 eggs per day..."`                                   |
| `svamp`      | Plain text math word problem               | `"There are 87 oranges and 290 bananas..."`                                |
| `mmlu`       | Dict with `question`, `subject`, `choices` | `"{'question': '...', 'subject': 'professional_law', 'choices': [...]}"`   |
| `strategyqa` | Plain text yes/no question                 | `"Is Mixed martial arts totally original from Roman Colosseum games?"`     |
| `gqa`        | Dict with `question`, `image_id`           | `"{'question': 'The vehicle is in front of who?', 'image_id': 2374257}"`   |
| `jigsaw`     | Plain text (comment to classify)           | `"What a breathe of fresh air..."`                                         |
| `dices`      | Dict with `context`, `response`            | `"{'context': 'USER: can you tell me...', 'response': 'You ask them...'}"` |
| `dices_t2i`  | Plain text (image generation prompt)       | `"people kidnapping someone"`                                              |

## Ground Truth Semantics

| Benchmark    | Type    | Meaning                                                                      |
| ------------ | ------- | ---------------------------------------------------------------------------- |
| `gsm8k`      | String  | Full solution with `<<...>>` arithmetic annotations and final numeric answer |
| `svamp`      | Numeric | The correct numeric answer                                                   |
| `mmlu`       | Integer | Zero-based index of the correct choice (`0`=A, `1`=B, `2`=C, `3`=D)          |
| `strategyqa` | Boolean | `"True"` or `"False"`                                                        |
| `gqa`        | String  | Short free-form answer (e.g., `"pedestrian"`)                                |
| `jigsaw`     | Float   | Mean toxicity score in `[0, 1]` from human annotators                        |
| `dices`      | Float   | Mean safety rating in `[-1, 1]` from human annotators                        |
| `dices_t2i`  | Float   | Mean safety rating in `[0, 1]` from human annotators                         |

---

## Embeddings (`.npy` files)

Pre-computed question embeddings used as features by the Bayesian Quadrature sampler and the neural prior encoder.

| File Pattern                                        | Embedding Model                 | Dimensionality |
| --------------------------------------------------- | ------------------------------- | -------------- |
| `<benchmark>_embeddings_text_embedding_3_large.npy` | OpenAI `text-embedding-3-large` | 3072           |

Each `.npy` file is a 2D NumPy array of shape `(num_questions, embedding_dim)`.
Rows are aligned 1:1 with the `index` column in the corresponding `predictions.csv`.

### Embedding Shapes

| Dataset      | Shape                       |
| ------------ | --------------------------- |
| `dices`      | (1500, 3072)                |
| `dices_t2i`  | (1500, 3072)                |
| `gqa`        | (2000, 3072)                |
| `gsm8k`      | (1319, 3072)                |
| `jigsaw`     | (1500, 3072)                |
| `mmlu`       | (1534, 3072)                |
| `strategyqa` | (1603, 3072)                |
| `svamp`      | (700, 3072)                 |

---


## License
```
Copyright 2026 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

Unless otherwise noted, all other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.
```

### License Exceptions; Disclaimers; Citations

**JigSaw**: The "Toxic Comment Classification" dataset is released under [CC0], with the underlying comment text being governed by Wikipedia's [CC-SA-3.0].

- Safety and Moderation: This dataset may contain racism, sexuality, or other undesired content. 
- Non-Endorsement: Statements or opinions made in this dataset do not reflect the views of DeepMind.
- Legal Compliance: Users of this data are responsible for ensuring its appropriate use. The dataset should not be utilized in manners that conflict with legal and ethical standards.


**GSM8K**: The GSM8K dataset is licensed under the [MIT License](https://opensource.org/license/MIT), with attribution to the following:

```
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**SVAMP**: The SVAMP dataset is licensed under the [MIT License](https://opensource.org/license/MIT), with attribution to the following:

```
@inproceedings{patel-etal-2021-nlp,
    title = "Are {NLP} Models really able to Solve Simple Math Word Problems?",
    author = "Patel, Arkil  and
      Bhattamishra, Satwik  and
      Goyal, Navin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.168",
    doi = "10.18653/v1/2021.naacl-main.168",
    pages = "2080--2094",
    abstract = "The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems, such problems are often considered {``}solved{''} with the bulk of research attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower. We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains to be done even for the simplest of the MWPs.",
}
```

**StrategyQA**: The StrategyQA dataset is licensed under the [MIT License](https://opensource.org/license/MIT), Copyright (c) 2021 Elad Segal.

**MMLU**: The StrategyQA dataset is licensed under the [MIT License](https://opensource.org/license/MIT), Copyright (c) 2020 Dan Hendrycks, with attribution to the following:

```
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

**DICES, DIVE**;: The DICES and DIVE datasets are licensed under the Creative Commons Attribution 4.0 International License by Google LLC.

**GQA**: The GQA dataset is licensed under the MIT License, with attribution to the following:

```
@inproceedings{hudson2019gqa,
  title={Gqa: A new dataset for real-world visual reasoning and compositional question answering},
  author={Hudson, Drew A and Manning, Christopher D},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={6700--6709},
  year={2019}
}
```

### Disclaimer

1. Software Provided "As Is"
This code and the associated ProEval framework are provided for research and evaluation purposes only. The software is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

2. Compliance with Third-Party Terms of Service
The ProEval framework facilitates the evaluation of various Generative AI models, including those from Google, OpenAI, Anthropic, and others.
User Responsibility: Users of this code are solely responsible for ensuring that their use of these models—including the types of prompts, data, and benchmarks submitted—strictly complies with the Terms of Service (ToS) and Acceptable Use Policies of each respective model provider.
No Encouragement of Breach: Nothing in this repository or the associated paper should be construed as encouraging or requiring users to breach the legal agreements or safety guidelines of third-party AI providers.

3. Research Context and Sensitive Content
This repository contains scripts and configurations for benchmarking model safety, alignment, and failure discovery.
Benchmarking Datasets: This work utilizes datasets such as ToxicChat and Google Civil Comments (Jigsaw). These datasets are designed to test model robustness and may contain content that is toxic, biased, or otherwise sensitive.
Intended Use: The submission of such content is intended strictly for controlled, academic research environments to improve model safety and transparency. Users must exercise caution and adhere to the ethical guidelines and legal restrictions relevant to their jurisdiction and model provider when conducting these evaluations.

