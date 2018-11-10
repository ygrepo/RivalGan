#!/bin/bash
echo "Training SVM"


python -W ignore insight/ai/pipeline.py --train_classifier true
