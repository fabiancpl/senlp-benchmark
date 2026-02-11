#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# v3 - lr 1e-4
# v4 - lr 1e-5

DATASET=requirement_completion
VERSION=v4
TASK_ARGS="--version $VERSION"
TRAINING_ARGS="--epochs 50 --batch_size_per_gpu 16 --gradient_accumulation_steps 1 --learning_rate 1e-5"


export WANDB_PROJECT="senlp_finetuning"
export WANDB_TAGS=$DATASET,$VERSION


# BERT

# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# RoBERTa

# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# ModernBERT

# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-base \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-large \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# CodeBERT

# deepspeed finetune_lm_mlm.py \
#     --dataset_name $DATASET \
#     --base_model_name codebert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS
