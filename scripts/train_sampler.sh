#!/bin/bash
echo "Training SMOTE"


python -W ignore insight/ai/pipeline.py --train_classifier true
