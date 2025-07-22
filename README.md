# Shakespearean Text Generation with Fine-Tuned GPT-2

## Introduction
This project demonstrates how to fine-tune a pre-trained **GPT-2 (Generative Pre-trained Transformer 2)** model to generate text in a specific style—in this case, the style of William Shakespeare. By leveraging a powerful, general-purpose language model and adapting it to a specialized dataset, we can create a text generator that mimics the vocabulary, syntax, and tone of Shakespearean writing.

The project uses the **Hugging Face `transformers`** library, which provides the necessary tools for downloading, configuring, and training state-of-the-art NLP models with PyTorch.

## 🫡 You can help me by Donating
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/heytanix)

### Dataset Overview:
- **Dataset**: A single text file (`shake.txt`) containing the complete works of William Shakespeare.
- **Source**: This text serves as the training corpus for fine-tuning.
- **Format**: Plain text.

**Objective**:
The primary goal is to adapt the general English knowledge of the pre-trained GPT-2 model to the specific nuances of Shakespearean English. The final model should be able to generate coherent and stylistically appropriate text from a given prompt.

## Core Technologies & Libraries Used
The implementation relies on the following key libraries and frameworks:

1.  **PyTorch**: The core deep learning framework used for model training and computation.
2.  **Hugging Face `transformers`**: Provides the pre-trained GPT-2 model (`GPT2LMHeadModel`), tokenizer (`GPT2Tokenizer`), and the high-level `Trainer` API for simplifying the training loop.
3.  **Hugging Face `accelerate`**: Used under the hood by the `Trainer` to enable seamless hardware acceleration on GPUs.
4.  **Hugging Face `datasets`**: Used for efficient data handling and preparation.

## Project Workflow
The project is structured into a series of clear, sequential steps as demonstrated in the Jupyter Notebook.

### Step 1: Environment Setup
- Installed all necessary libraries, including `torch` and the Hugging Face ecosystem.
- Verified the availability of a CUDA-enabled GPU to accelerate the training process.

### Step 2: Model and Data Preparation
- Loaded the pre-trained `gpt2` model and its corresponding tokenizer from the Hugging Face Hub.
- Handled a technical requirement for GPT-2 by setting the `pad_token` to the `eos_token` (end-of-sequence) to allow for dynamic batching.
- Loaded the `shake.txt` file and processed it using the `TextDataset` and `DataCollatorForLanguageModeling` classes, which prepared the text for causal language modeling by tokenizing and chunking it into appropriate block sizes.

### Step 3: Model Fine-Tuning
- Configured all training settings using the `TrainingArguments` class. Key parameters included the number of epochs, batch size, save steps, and enabling **FP16 (mixed-precision) training** for better performance and lower memory usage.
- Initialized the `Trainer` object with the model, training arguments, and prepared dataset.
- Started the fine-tuning process by calling `trainer.train()`. The model's weights were updated over 5 epochs to minimize the prediction loss on the Shakespearean text.

### Step 4: Saving and Inference
- Saved the final fine-tuned model weights and tokenizer configuration to disk. This allows for easy reuse without retraining.
- Created a `text-generation` pipeline using the fine-tuned model to perform inference.
- Generated new text from a simple prompt ("Shakespeare Quote") to demonstrate the model's newly acquired stylistic capabilities.

### Conclusion
This project successfully demonstrates the power of transfer learning in NLP. By starting with a pre-trained GPT-2 model, we were able to achieve impressive stylistic adaptation with a relatively straightforward fine-tuning process. The **Hugging Face `Trainer` API** significantly simplifies the training loop, making such projects highly accessible.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for their incredible `transformers` library and open-source models.
- The providers of the public domain Shakespearean texts (e.g., Project Gutenberg).
- The PyTorch team for developing and maintaining the core deep learning framework.
