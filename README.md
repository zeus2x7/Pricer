# Pricer

Pricer is a project that demonstrates fine-tuning a large language model (Meta-Llama 3.1-8B) using QLoRA techniques to predict item prices based on textual descriptions. The project leverages custom data processing, efficient LoRA-based fine-tuning, and evaluation scripts to produce a model that estimates prices from item details.

## Features

- **QLoRA Fine-Tuning:** Uses LoRA (Low-Rank Adaptation) to fine-tune the Meta-Llama 3.1-8B model efficiently.
- **Custom Data Processing:** Implements data cleaning and prompt generation for price prediction using custom modules.
- **Parallel Data Loading:** Uses parallel processing to load and preprocess data from the Amazon Reviews dataset.
- **Model Evaluation:** Provides testing scripts to assess model performance and visualize prediction errors.
- **HuggingFace Hub Integration:** Pushes the fine-tuned model to the HuggingFace Hub for easy sharing and deployment.
- **Weights & Biases Tracking (Optional):** Integrates with Weights & Biases for experiment tracking and monitoring.

## Project Structure

- **`qlora_llama_3_1_finetune.py`**  
  Main script for fine-tuning the Llama 3.1 model using QLoRA. It configures the training parameters, LoRA settings, and dataset details, and pushes the trained model to the HuggingFace Hub.  
  :contentReference[oaicite:0]{index=0}

- **`items.py`**  
  Contains the `Item` class, which is responsible for cleaning raw item data, generating prompts, and preparing text for training.  
  :contentReference[oaicite:1]{index=1}

- **`loaders.py`**  
  Implements the `ItemLoader` class that loads and processes the dataset in parallel from the "McAuley-Lab/Amazon-Reviews-2023" dataset.  
  :contentReference[oaicite:2]{index=2}

- **`tester.py`**  
  Provides tools for evaluating model performance. It includes functionality to compute errors, plot predictions versus ground truth, and generate reports based on the model’s estimates.  
  :contentReference[oaicite:3]{index=3}

- **Additional Notebooks:**  
  Other notebooks (e.g., for dataset generation, baseline models, testing unfine-tuned models) are included to support different stages of the development and evaluation pipeline.
