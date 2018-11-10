#!/bin/bash
echo "Generating scores for baseline classifier"


python -W ignore insight/ai/pipeline.py --classifier_scores true
