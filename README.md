# IMDD-1M: Towards Open-Vocabulary Industrial Defect Understanding

[![Dataset](https://img.shields.io/badge/Dataset-1.24M%20Images-green)](https://github.com/NinaNeon/IMDD-1M-Towards-Open-Vocabulary-Industrial-Defect-)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page-url)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> **Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset**  
>  2026 Submission #****

## Overview

IMDD-1M is the first large-scale Industrial Multimodal Defect Dataset comprising 1,240,379 aligned image-text pairs across 63 industrial domains and 421 defect types. We train a diffusion-based foundation model from scratch that achieves comparable performance with less than 5% of the training data required by conventional supervised methods.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | 1,240,379 |
| Normal Samples | 285,451 |
| Anomaly Samples | 954,928 |
| Industrial Domains | 63 |
| Defect Types | 421 |
| Image Resolution | 512×512 |
| Text Description Length | ~42 words |

## Architecture

Our framework consists of three main components:

### 1. Industrial Diffusion U-Net (860M parameters)
- Based on Stable Diffusion v1.5 architecture
- Trained from random initialization on IMDD-1M
- Four encoder blocks with channels [320, 640, 1280, 1280] at strides [1, 2, 4, 8]
- Decoder mirrors encoder with skip connections
- Cross-attention layers for text conditioning

### 2. Implicit Captioner (0.3M parameters)
- Frozen CLIP image encoder + trainable 2-layer MLP
- Projects 512-dim CLIP embeddings to 768-dim text embedding space
- Enables text conditioning when captions are unavailable
- Trained with stochastic conditioning (50% real captions, 50% implicit embeddings)

### 3. Mask2Former Generator (45M parameters)
- Pixel decoder with Feature Pyramid Network (FPN)
- Transformer decoder with 100 learnable queries
- Produces masks and embeddings for open-vocabulary classification

## Training Protocol

### Stage 1: Foundation Model Pretraining
```
Dataset: IMDD-1M (1,240,379 images)
Loss: L_diff + 0.3 × L_imp
Epochs: 100
Batch Size: 256
GPUs: 8× H100
Training Time: 72 hours
Optimizer: AdamW (lr=1e-4)
```

### Stage 2: Downstream Task Fine-tuning
```
Loss: L_mask + 0.5 × L_cls/ground
Epochs: 50
Batch Size: 16
GPUs: 8× H100
Training Time: 4 hours
Optimizer: AdamW (lr=5e-5)
```

## Performance

### Data Efficiency
Our model achieves 96.1% accuracy using only 200 samples per class, requiring less than 5% of the training data compared to conventional supervised methods (~4,000 samples per class).

## Installation

```bash
git clone https://github.com/yourusername/IMDD-1M.git
cd IMDD-1M

conda create -n imdd python=3.9
conda activate imdd

pip install -r requirements.txt
```

## Dataset Structure

```
IMDD-1M/
├── images/
│   ├── semiconductor/
│   ├── steel_processing/
│   ├── electronics/
│   └── ...
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── metadata.json
```

## Quick Start

### 1. Text-Guided Defect Generation

```python
from models.diffusion import IndustrialDiffusionModel

model = IndustrialDiffusionModel.from_pretrained("checkpoints/imdd1m_diffusion.pth")
prompt = "metal surface with scratches and oxidation"
generated_image = model.generate(prompt, num_inference_steps=50)
```

### 2. Defect Classification

```python
from models.classifier import DefectClassifier

classifier = DefectClassifier.from_pretrained("checkpoints/imdd1m_classifier.pth")
image = load_image("path/to/defect_image.jpg")
prediction = classifier.predict(image)
print(f"Defect type: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")
```

### 3. Defect Segmentation

```python
from models.segmentation import DefectSegmenter

segmenter = DefectSegmenter.from_pretrained("checkpoints/imdd1m_segmenter.pth")
image = load_image("path/to/defect_image.jpg")
mask = segmenter.segment(image)
```

### 4. Fine-tuning on Custom Dataset

```python
from trainers.finetune import FineTuner

finetuner = FineTuner(
    pretrained_model="checkpoints/imdd1m_diffusion.pth",
    dataset_path="path/to/custom_dataset",
    num_samples_per_class=200
)

finetuner.train(epochs=50, batch_size=16)
```

## Repository Structure

```
IMDD-1M/
├── models/
│   ├── config.json
│   ├── merges.txt
│   ├── model_index.json
│   ├── preprocessor_config.json
│   ├── readme.md
│   ├── scheduler_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── third_party/
│   └── ODISE/
│       ├── .gitignore
│       ├── GETTING_STARTED.md
│       ├── LICENSE
│       ├── MANIFEST.in
│       ├── README.md
│       ├── setup.cfg
│       └── setup.py
├── LICENSE
├── Object_detection.py
├── README.md
├── classify.py
├── integrate_custom_unet_vae.py
└── requirements.txt
```

## Comparison with Existing Datasets

| Dataset | Year | # Images | # Domains | Text Annotations |
|---------|------|----------|-----------|------------------|
| DAGM | 2016 | 1.5K | 1 | No |
| KolektorSDD | 2019 | 400 | 1 | No |
| MVTec AD | 2019 | 5.4K | 15 | No |
| BTAD | 2021 | 2.5K | 3 | No |
| VisA | 2022 | 10.8K | 12 | No |
| Real-IAD | 2024 | 67K | 30 | No |
| **IMDD-1M** | 2025 | **1.24M** | **63** | **Yes** |

## Key Features

### Expert-Verified Annotations
All annotations verified by domain experts with structured templates including product category, material composition, defect type, spatial location, and root causes.

### Diverse Industrial Coverage
- Semiconductor: Wafer defects, contamination, pattern failures
- Electronics: Solder defects, PCB anomalies, component issues
- Metals: Surface pitting, corrosion, cracks, delamination
- Textiles: Fabric defects, staining, pattern irregularities
- Packaging: Container defects, labeling errors, damage

### Multimodal Learning
Aligned image-text pairs enable vision-language understanding with fine-grained textual descriptions containing morphological details and specialized industrial terminology.

## Implementation Details

- Total parameters: 890M (860M U-Net + 0.3M Implicit Captioner + 45M Mask Generator)
- Inference time: 0.35s per image on A100 GPU
- Feature extraction: Single forward pass at t=50 timestep
- Frozen VAE for 8× compression
- Frozen CLIP text encoder (768-dim)

## Citation

```bibtex
@inproceedings{imdd1m2026,
  title={Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset},
  author={Anonymous},
  booktitle={****},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank our industrial partners in petrochemical, metal processing, and powder metallurgy sectors for their data contributions. We acknowledge the public datasets that formed the foundation of our work: DAGM, MVTec AD, KolektorSDD, VisA, and BTAD.
