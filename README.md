# Mental Health Text Classification Using Transformer Models

## Project Overview
This project investigates the use of transformer-based language models for multi-class mental health classification using Reddit text data. The objective is to evaluate and compare the performance of general purpose and domain specific transformers in detecting mental health related language patterns, with a focus on anxiety and depression.

The project is part of my BSc Data Science and Analytics Final Year Project and contributes to research on applying NLP techniques to mental health data in a responsible and ethical manner.


## Models Implemented
The following transformer architectures are implemented and evaluated:

- DistilBERT – used  due to its reduced size and faster training time, enabling efficient experimentation and hyperparameter tuning
- RoBERTa – a larger general-purpose transformer for performance comparison
- MentalBERT – a domain-specific transformer pretrained on mental health-related text

All models are fine-tuned using the Hugging Face Transformers framework.

## Dataset
The dataset consists of Reddit posts labelled into multiple mental health-related classes. The data is used strictly for academic research purposes.

⚠️ Raw data is not included in this repository to ensure privacy and ethical compliance.

## Methodology Summary
- Text preprocessing and tokenisation using transformer tokenisers
- Fine-tuning pretrained transformer models on labelled Reddit data
- Evaluation on a held-out test set using standard classification metrics
- Model comparison based on performance, computational cost and training efficiency
- Error analysis focusing on false positives, false negatives and overlapping linguistic patterns between anxiety and depression

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

All models are evaluated on the same unseen test dataset to ensure a fair and valid comparison.

## Repository Contents
- `mental_health_ transformer_classification.ipynb`  
  Main notebook containing data preprocessing, model fine-tuning, evaluation, comparison and error analysis.

