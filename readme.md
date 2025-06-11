# Jigsaw Toxicity Classifier Fine-tuning
This repository contains a Google Colab notebook that demonstrates the process of fine-tuning a pre-trained DistilBERT model for toxicity classification using the Jigsaw Toxicity Raw Dataset. The notebook covers hyperparameter optimization, model training, evaluation, and inference.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ITHXGQKGuZFwiAASyiw9GYXkewM8tzwD?usp=sharing)

[Check out the Model and Training Results here](https://huggingface.co/godsofheaven/final_model_toxicity_classification)

**1. Project Description:**
This Google Colab notebook provides a complete workflow for building a comment text toxicity classifier. It leverages the power of transfer learning by fine-tuning a pre-trained DistilBERT model. The process includes:

Loading and preprocessing the Jigsaw Toxicity Raw Dataset.
Splitting data into dedicated training, validation, and test sets to ensure robust evaluation.
Performing hyperparameter search using optuna and Hugging Face Trainer to find optimal training configurations.
Training the final model with the best identified hyperparameters.
Evaluating the model's performance on a held out test set to get an unbiased generalization estimate.
Demonstrating how to perform inference (make predictions) on new text samples.

**2. Dataset:**
This project uses the [Jigsaw Toxicity Pred Dataset](https://huggingface.co/datasets/google/jigsaw_toxicity_pred), available through the Hugging Face datasets library.

Description: Contains text comments labeled for toxic, severe_toxic, obscene, threat, insult and identity_hate.
Splits: The original dataset provides train and test splits. In this notebook, the original test split is further divided into a validation set (for hyperparameter tuning) and a new, smaller true test set (for final, unbiased evaluation).

**3. Model Architecture**
The model fine-tuned in this notebook is DistilBERT, a distilled version of BERT.

Base Model: distilbert-base-uncased
Task: Sequence Classification (binary: safe/unsafe)
Source: Hugging Face transformers library.

**4. Setup and Requirements**
This notebook runs optimally in Google Colab due to its free GPU access and pre-installed libraries.

Prerequisites:
Key Libraries (installed via pip install commands in the notebook):

- transformers (Hugging Face)
- datasets (Hugging Face)
- evaluate (Hugging Face)
- optuna (for hyperparameter optimization)
- accelerate (for distributed training features, often a dependency of transformers)
- torch (PyTorch)
- fsspec
- kaggle (for fetching the dataset)

**5. Evaluation & Results**
The hyperparameter search uses F1-score on the validation set as its primary objective. The final model's performance on the held out test set will be printed at the end of the evaluation section.

Expected metrics include:

- eval_f1
- eval_accuracy
- eval_loss
- eval_runtime
- eval_samples_per_second
- eval_steps_per_second
- epoch


The exact values will depend on the hyperparameter search results and the training process.

**6. Inference**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCS4SAF2ET9s9ZHTk-Yk1DtwWZfm_sY2?usp=sharing)

```
from transformers import pipeline

pipe = pipeline("text-classification", model="godsofheaven/final_model_toxicity_classification")
user_input = "Enter your comment here"
prediction = pipe(user_input)
pred = prediction[0]

# Interpret the prediction
predicted_label_index = int(pred['label'].split('_')[1])
predicted_class = "UNSAFE" if predicted_label_index == 1 else "SAFE"
confidence = pred['score']

print("\n--- Pipeline Prediction for Single Input ---")
print(f"Text: '{user_input}'")
print(f"  Prediction: {predicted_class} (Confidence: {confidence:.4f})")
print("-" * 30)
```
