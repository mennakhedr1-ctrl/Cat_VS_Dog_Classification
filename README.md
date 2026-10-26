# Cat_VS_Dog_Classification
Dogs vs. Cats image classification with a two-stage fine-tuned MobileNetV2 (99% test accuracy).

## Overview
This project fine-tunes a pretrained MobileNetV2 to tell cats from dogs. Images and labels are built together from the same file list, so they always match. Augmentation (flip, rotation, zoom), dropout, early stopping .

## Approach
1. Resized images to 224x224 RGB and applied MobileNetV2 `preprocess_input`
2. Stage 1: froze the base model and trained a new classification head
3. Stage 2: unfroze the last 30 layers and fine-tuned with Adam at 1e-5
4. Evaluated on the held-out test set

## Results
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| dog | 0.983 | 0.998 | 0.990 | 400 |
| cat | 0.997 | 0.983 | 0.990 | 400 |

| Metric | Score |
|--------|-------|
| Test accuracy | 99.0% |
| Macro avg F1 | 0.990 |

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, NumPy, Pillow, Matplotlib
