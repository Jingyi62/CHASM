<!-- @format -->

# API Inference for Advertisement Detection

This repository contains a Python script for detecting covert advertisements in social media posts using OpenAI API models.

## Overview

The `api_inference.py` script analyzes social media posts from the "CHASM-Covert_Advertisement_on_RedNote" dataset to determine whether they contain covert advertisements. It uses OpenAI's models (like GPT-4) to classify posts as either advertisements (1) or non-advertisements (0).

## Features

- Loads and processes data from a Hugging Face dataset
- Supports multimodal analysis (text and images)
- Performs inference using OpenAI API models
- Calculates and reports evaluation metrics (accuracy, precision, recall, F1 score)
- Saves results to CSV and metrics to JSON

## Requirements

- Python 3.6+
- Required packages:
  - datasets
  - openai
  - huggingface_hub

## Setup

1. Install the required packages:

   ```
   pip install datasets openai huggingface_hub
   ```

2. Set your Hugging Face token as an environment variable (optional):

   ```
   export HF_TOKEN=your_huggingface_token
   ```

3. Configure your OpenAI API key and base URL in the script or through environment variables.

## Usage

Run the script:

```
python api_inference.py
```

The script will:

1. Prompt you to log in to Hugging Face (if not already authenticated)
2. Ask you to select a model for inference
3. Allow you to set a sampling ratio and output filename
4. Run inference on the selected samples
5. Save results and evaluation metrics

## Output

- `api_inference_results.csv`: Contains the inference results for each sample
- `api_inference_metrics.json`: Contains evaluation metrics (accuracy, precision, recall, F1)

## Model Selection

The script supports multiple OpenAI models:

- GPT-4
- GPT-4o-2024-08-06
- GPT-4o-mini
- GPT-3.5-turbo

## Advertisement Detection Criteria

The model analyzes posts based on these characteristics of covert advertisements:

1. Clear promotional evidence (purchase links, sales instructions)
2. Promotional language style (clickbait titles, sales pitches)
3. Focus on specific products or brands
