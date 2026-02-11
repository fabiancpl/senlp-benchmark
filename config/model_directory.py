"""Model directory.

Here is detailed all the models from HuggingFace used during the pre-training or fine-tuning stages.
Some relevant attributes like case-sensitiveness are included.
"""

hf_models = {
    "bert-base": {
        "org_id": "google-bert",
        "model_id": "bert-base-cased",
        "architecture": "encoder",
        "mlm_probability": 0.15
    },
    "bert-large": {
        "org_id": "google-bert",
        "model_id": "bert-large-cased",
        "architecture": "encoder",
        "mlm_probability": 0.15
    },
    "roberta-base": {
        "org_id": "FacebookAI",
        "model_id": "roberta-base",
        "architecture": "encoder",
        "mlm_probability": 0.15
    },
    "roberta-large": {
        "org_id": "FacebookAI",
        "model_id": "roberta-large",
        "architecture": "encoder",
        "mlm_probability": 0.15
    },
    "modernbert-base": {
        "org_id": "answerdotai",
        "model_id": "ModernBERT-base",
        "architecture": "encoder",
        "mlm_probability": 0.3
    },
    "modernbert-large": {
        "org_id": "answerdotai",
        "model_id": "ModernBERT-large",
        "architecture": "encoder",
        "mlm_probability": 0.3
    },
    "codebert-base": {
        "org_id": "microsoft",
        "model_id": "codebert-base-mlm",
        "architecture": "encoder",
        "mlm_probability": 0.15
    },
    "gpt2-small": {
        "org_id": "openai-community",
        "model_id": "gpt2",
        "architecture": "decoder"
    },
    "gpt2-medium": {
        "org_id": "openai-community",
        "model_id": "gpt2-medium",
        "architecture": "decoder"
    },
    "gpt2-large": {
        "org_id": "openai-community",
        "model_id": "gpt2-large",
        "architecture": "decoder"
    },
    "gpt2-xl": {
        "org_id": "openai-community",
        "model_id": "gpt2-xl",
        "architecture": "decoder"
    },
    "llama3.2-1b": {
        "org_id": "meta-llama",
        "model_id": "Llama-3.2-1B",
        "architecture": "decoder"
    },
    "llama3.2-3b": {
        "org_id": "meta-llama",
        "model_id": "Llama-3.2-3B",
        "architecture": "decoder"
    },
    "codellama-7b": {
        "org_id": "meta-llama",
        "model_id": "CodeLlama-7b-hf",
        "architecture": "decoder"
    },
    "starcoder2-3b": {
        "org_id": "bigcode",
        "model_id": "starcoder2-3b",
        "architecture": "decoder"
    },
    "starcoder2-7b": {
        "org_id": "bigcode",
        "model_id": "starcoder2-7b",
        "architecture": "decoder"
    },
    "t5-small": {
        "org_id": "google-t5",
        "model_id": "t5-small",
        "architecture": "encoder-decoder"
    },
    "t5-base": {
        "org_id": "google-t5",
        "model_id": "t5-base",
        "architecture": "encoder-decoder"
    },
    "t5-large": {
        "org_id": "google-t5",
        "model_id": "t5-large",
        "architecture": "encoder-decoder"
    },
    "t5-3b": {
        "org_id": "google-t5",
        "model_id": "t5-3b",
        "architecture": "encoder-decoder"
    },
    "codet5p-220m": {
        "org_id": "Salesforce",
        "model_id": "codet5p-220m",
        "architecture": "encoder-decoder"
    },
    "codet5p-770m": {
        "org_id": "Salesforce",
        "model_id": "codet5p-770m",
        "architecture": "encoder-decoder"
    }
}
