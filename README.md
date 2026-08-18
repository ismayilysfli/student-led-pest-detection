# AZira Pest Detection Pipeline

This repository contains the public training and deployment pipeline used for the pest-detection component of **AZira**, an AI-powered mobile application developed by Guardian-X.

AZira allows users to capture or upload an image of a pest and receive AI-based pest identification together with supporting information such as affected crops, treatment considerations, and preventive guidance.

The application is publicly available on Google Play.

## Project Overview

The pest-detection model was developed as a multi-class image classification system using approximately **18,000 images across 15 pest classes**.

The pipeline covers:

- dataset preparation and validation
- corrupted-image filtering
- model training and evaluation
- transfer learning with FastViT
- ONNX export
- TensorFlow Lite conversion for mobile integration

The model was trained using **PyTorch** and a pretrained **FastViT-T12** architecture from `timm`.

After training, the model was converted through the following deployment pipeline:

**PyTorch → ONNX → TensorFlow Lite**

This allowed the trained model to be prepared for use inside the AZira mobile application.

## Repository Structure

### `prepare_fastvit_dataset.py`

Prepares the image dataset for training.

The script handles dataset organization and filtering required before model training.

### `train_fastvit_t12.py`

Trains and evaluates the FastViT-T12 image classification model using PyTorch and `timm`.

The training pipeline includes transfer learning, validation, and test evaluation.

### `export_fastvit_onnx.py`

Exports the trained PyTorch model to ONNX format for deployment and interoperability.

### `export_fastvit_tflite.py`

Converts the exported model toward TensorFlow Lite format for mobile inference.

During deployment testing, a float16 conversion showed unreliable behavior, so the working float32 version was retained for integration.

### `requirements_fastvit.txt`

Contains the Python dependencies required for the training and export pipeline.

## Tech Stack

- Python
- PyTorch
- timm
- FastViT-T12
- ONNX
- TensorFlow Lite
- NumPy
- image classification
- transfer learning

## Data and Model Artifacts

The dataset, trained model weights, detailed evaluation results, and other internal artifacts are **not included in this public repository** because they are part of Guardian-X's private startup work.

This repository is intended to provide the public implementation of the training and deployment pipeline without exposing proprietary data or internal results.

## My Contribution

I worked on the AI pipeline for AZira, including:

- dataset preparation and quality checking
- model training and evaluation
- FastViT-based transfer learning
- deployment conversion from PyTorch to ONNX and TensorFlow Lite
- debugging model conversion issues
- preparing the model for mobile integration

I also contributed to the supporting pest knowledge base used by the application.

## Related Links

**Guardian-X:**  
https://guardian-x-new.vercel.app/

**AZira on Google Play:**  
https://play.google.com/store/apps/details?id=com.jamalzadeh.azira

**GitHub Profile:**  
https://github.com/ismayilysfli

## Note

This repository contains only the public technical portion of the project. Some implementation details, datasets, model artifacts, and evaluation results remain private to Guardian-X.
