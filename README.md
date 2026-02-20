## SELU: A Software Engineering Language Understanding Benchmark

Source code to replicate the results reported in *F. Peña, S. Herbold, "SELU: A Software Engineering Language Understanding Benchmark," 2026*.

### :memo: Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in code understanding and generation. However, their effectiveness on non‐code Software Engineering (SE) tasks remains underexplored. We present `Software Engineering Language Understanding' (SELU), the first comprehensive benchmark for evaluating LLMs on 22 SE textual artifacts NLU tasks, spanning from identifying whether a requirement is functional or non-functional to estimating the effort required to implement a development task. SELU covers classification, regression, Named Entity Recognition (NER), and Masked Language Modeling (MLM) tasks, with data drawn from diverse sources such as issue tracking systems and developer forums. We fine-tune 22 open-source LLMs, both generalist and domain-adapted; and prompt two proprietary alternatives using zero-shot a 3-shot prompting strategies. Performance is measured using metrics such as F1-macro, SMAPE, F1-micro, and accuracy, and compared via the Bayesian signed-rank test. Our results show that fine-tuned models across various sizes and architectures perform best, exhibiting high mean performance and low across-task variance. Furthermore, domain adaptation via code-focused pre-training does not yield significant improvements and might even be counterproductive for developer communication tasks.

### :card_index_dividers: Repository organization

- `datasets`: Inventory of the 22 non-code SE tasks included in SELU and their respective datasets.
- `evaluation`: Scripts to fine-tune/prompt and evaluate the models on the different tasks.
- `preprocessing`: Scripts to prepare the data previous to splitting and tokenization.
- `utils`: Common functions used during pre-processing and evaluation.

### :gear: Setup

All our experiments are run on a server with 8 NVIDIA A100 GPUs.

### :bookmark: Cite as

```
@misc{peña2026selu,
  title={SELU: A Software Engineering Language Understanding Benchmark}, 
  author={Fabian C. Peña and Steffen Herbold},
  year={2026},
  eprint={2506.10833v2},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2506.10833v2}, 
}
```
