# Amazon ML Challenge 2025: Multi-Modal Price Prediction
###### Team: Amir Hamza Khan, Moh Ahmad, Taiyabullah, Mohd Amir Pasha

##### Score Achieved: SMAPE ≈ 54.6

## Overview
This repository contains the codebase for our end-to-end solution of second model submission to the Amazon ML Challenge 2025. The task was to predict product prices using multi-modal data (text, images) from the Amazon dataset. Our approach fuses powerful text and image embeddings from recent models (Gemma, CLIP) and applies a supervised neural network to regression. Read below to understand our technical stack, workflow, and achieved metrics. Our initial model was still undergoing final training at the close of the competition due to device constraints; as a result, this secondary model became our official entry.

## Challenge Description
Predict price given:

Product description text

Product image

Objective metric: Symmetric Mean Absolute Percentage Error (SMAPE)

Goal: Build a robust, scalable ML pipeline to efficiently leverage cross-modal signals for price prediction.

## Process Map & Workflow
Below is a simplified flowchart of our core pipeline:

```
                   ┌───────────────────────┐
                   │    Data Collection    │
                   │(images, texts, prices)│
                   └──────────┬────────────┘
                              │
                ┌─────────────▼────────────┐
                │      Preprocessing       │
                │- Clean text (HTML/tags)  │
                │- Validate images         │
                └────────────┬─────────────┘
                             │
         ┌──────────────┬────▼───────┬─────────────┐
         │              │            │             │
┌────────▼───┐   ┌──────▼─────┐  ┌───▼────────┐    │
│ Gemma      │   │  CLIP      │  │  Feature   │    │
│ Text Embed │   │ Text/Image │  │ Extraction │    │
└──────┬─────┘   └─────┬──────┘  └──────┬─────┘    │
       │               │                │          │
       └───────────────┴────────────┬───┘          │
                                    │              │
                        ┌───────────▼─────────────┐
                        │   Feature Fusion        │
                        │(Concatenate all embeds) │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │   Model Training        │
                        │ (SimpleNN - PyTorch)    │
                        └───┬─────────┬───────────┘
                            │         │
                    ┌───────▼──┐   ┌──▼────────┐
                    │Validation│   │ Hyperparam│
                    │(SMAPE,   │   │ Tuning    │
                    │R², MAE,  │   │           │
                    │RMSE)     │   └───────────┘
                    └─────┬────┘
                          │
              ┌───────────▼───────────┐
              │ Prediction & Output   │
              │ (Test set, CSVs)      │
              └───────────────────────┘


```

#### Project Workflow: Amazon ML Challenge 2025
Stepwise Breakdown
1. Data Collection
Import image URLs, product descriptions, and target prices from Amazon datasets.

Download and validate images.

2. Preprocessing
Clean product descriptions (HTML tag stripping, emoji/unicode removal via BeautifulSoup & regex).

Prepare text for both CLIP and Gemma embedding models.

Images checked/converted to RGB format.

3. Embedding Generation
Text Embedding (Gemma):

SentenceTransformer with Gemma-300M model.

Handles long text via custom chunking and average pooling.

Image Embedding (CLIP):

OpenCLIP (ViT-L-14, laion2B-s32B-b82K) for image and text modality.

Features normalized, tokenized, batched.

Output: For each sample: [Gemma Embeddings | CLIP Text Embeddings | CLIP Image Embeddings]

4. Feature Fusion & Engineering
All embeddings for each product concatenated horizontally to build multi-modal feature vectors.

Train/validation split, features normalized using StandardScaler.

5. Model Architecture
SimpleNN (PyTorch)
Input: Fused embeddings (2304 units: 768×3)

Layers:

[Linear → BatchNorm → Dropout] × 5

Output: Single price prediction

Training: Adam optimizer, Early stopping, Metric logging.

python
class SimpleNN(nn.Module):
    # (See training_simplenn.ipynb for details)
6. Evaluation
Primary metric: SMAPE

Additional: R², MAE, RMSE

Best Validation SMAPE: 54.6

Training monitored with all metrics per epoch for transparency.

7. Prediction & Output
Model and scaler checkpoint bundled for reproducibility.

Predictions on holdout test set written to CSV.

Results
Score: SMAPE ≈ 54.6 (final validation, competitive for the challenge)

Model: SimpleNN on fused Gemma + CLIP embeddings

Technical Stack: PyTorch, SentenceTransformers, OpenCLIP, pandas, scikit-learn, tqdm

How to Run
Install dependencies (pip install -r requirements.txt)

Download images via image_download.ipynb

Run embedding_generation.ipynb to generate all embeddings

Train & evaluate using training_simplenn.ipynb

Final predictions with test-output.ipynb

Code Structure
utils.py — Download helpers

image_download.ipynb — Parallel image download

embedding_generation.ipynb — Embedding pipeline

training_simplenn.ipynb — Model training, SMAPE calculation, logging

test-output.ipynb — Bundled model inference on test set
