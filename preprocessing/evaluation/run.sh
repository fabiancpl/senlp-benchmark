#!/bin/bash
set -e


# Binary classification tasks
python preprocess.py --dataset_name bug_issue
python preprocess.py --dataset_name functional_requirement
python preprocess.py --dataset_name incivility
python preprocess.py --dataset_name quality_requirement
python preprocess.py --dataset_name safety_issue
python preprocess.py --dataset_name security_requirement
python preprocess.py --dataset_name tone_bearing


# Multi-class classification tasks
python preprocess.py --dataset_name closed_question
python preprocess.py --dataset_name commit_intent
python preprocess.py --dataset_name issue_intention
python preprocess.py --dataset_name issue_type
python preprocess.py --dataset_name question_quality
python preprocess.py --dataset_name review_type
python preprocess.py --dataset_name sentiment


# Multi-label classification tasks
python preprocess.py --dataset_name comment_type --variant java
python preprocess.py --dataset_name comment_type --variant pharo
python preprocess.py --dataset_name comment_type --variant python
python preprocess.py --dataset_name review_aspect
python preprocess.py --dataset_name smell_doc


# Regression tasks
python preprocess.py --dataset_name story_points
