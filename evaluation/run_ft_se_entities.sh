#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# v3 - lr 1e-4
# v4 - lr 1e-5

DATASET=se_entities
VERSION=v4
TASK_ARGS="--version $VERSION"
TRAINING_ARGS="--epochs 50 --batch_size_per_gpu 16 --gradient_accumulation_steps 1 --learning_rate 1e-5"
TRAINING_ARGS_8x2="--epochs 50 --batch_size_per_gpu 8 --gradient_accumulation_steps 2 --learning_rate 1e-5"


export WANDB_PROJECT="senlp_finetuning"
export WANDB_TAGS=$DATASET,$VERSION


# BERT

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# RoBERTa

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# ModernBERT

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# CodeBERT

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name codebert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# T5

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-small \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-base \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-large \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-3b \
#     --max_length 512 \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS



# CodeT5+

# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name codet5p-220m \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --not_use_safetensors


# deepspeed finetune_lm_ner.py \
#     --dataset_name $DATASET \
#     --base_model_name codet5p-770m \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --not_use_safetensors
