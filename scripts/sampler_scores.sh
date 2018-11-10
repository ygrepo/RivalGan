#!/bin/bash
echo "Generating scores for SMOTE sampler"


python -W ignore insight/ai/pipeline.py --classifier_scores true
