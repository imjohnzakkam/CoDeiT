# CoDeiT: Contrastive Data-efficient Transformers for Deepfake Detection

[![Paper]([https://img.shields.io/badge/paper-pdf-red](https://drive.google.com/file/d/1B8xb3C8t4y0FgPLDhflEPJqReOCwYFXb/view))](paper_link)

Official implementation of "CoDeiT: Contrastive Data-efficient Transformers for Deepfake Detection"

## Overview

CoDeiT is a novel framework for deepfake detection that leverages the strengths of hierarchical attention mechanism and contrastive learning. The model employs HiLo Attention to effectively disentangle and process high and low-frequency information, significantly enhancing its ability to detect subtle manipulations indicative of deepfakes.

### Key Features

- HiLo Transformer with hierarchical attention mechanism
- Contrastive learning framework for enhanced feature discrimination
- Efficient processing of high and low-frequency information
- State-of-the-art performance on multiple benchmark datasets
- Three model variants for different computational requirements

## Model Architecture

CoDeiT comes in three variants:

- **CoDeiT-S**: 22M parameters, 6 attention heads
- **CoDeiT-L**: 86M parameters, 12 attention heads
- **CoDeiT-XL**: 307M parameters, 24 attention heads

## Requirements

```
python>=3.8
torch>=2.0.0
timm
numpy
opencv-python
albumentations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/CoDeiT.git
cd CoDeiT

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
from codeit import CoDeiTModel, CoDeiTTrainer

# Initialize model
model = CoDeiTModel(variant='xl')  # Options: 's', 'l', 'xl'

# Initialize trainer
trainer = CoDeiTTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    num_epochs=100
)

# Start training
trainer.train()
```

### Inference

```python
import torch
from codeit import CoDeiTModel

# Load pretrained model
model = CoDeiTModel.from_pretrained('path/to/checkpoint')

# Perform inference
image = load_image('path/to/image')
prediction = model.predict(image)
```

## Datasets

The model has been evaluated on the following datasets:

- **Celeb-DF**
- **FaceForensics++**
- **Deepfake Detection Challenge (DFDC)**

## Results

### Performance on Benchmark Datasets

| Model      | DFDC (HQ)      | CelebDF (HQ)   |
|------------|----------------|----------------|
| CoDeiT-S   | 82.5% / 0.92   | 73.8% / 0.85   |
| CoDeiT-L   | 84.7% / 0.94   | 76.1% / 0.87   |
| CoDeiT-XL  | 86.9% / 0.95   | 78.5% / 0.89   |

*Format: Accuracy / AUC*

## Citation

```bibtex
@inproceedings{zakkam2024codeit,
  title={CoDeiT: Contrastive Data-efficient Transformers for Deepfake Detection},
  author={Zakkam, John and Jayaraman, Umarani and Rattani, Ajita and Sahayam, Subin},
  booktitle={},
  year={2024}
}
```

## Contributing

We welcome contributions to improve CoDeiT! Please feel free to submit issues, fork the repository and create pull requests for any improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Indian Institute of Information Technology Design and Manufacturing, Kancheepuram
- University of North Texas
- Shiv Nadar University
- The authors of FaceForensics++, Celeb-DF, and DFDC datasets

## Contact

John Zakkam - ced18i059@iiitdm.ac.in

Project Link: [https://github.com/username/CoDeiT](https://github.com/username/CoDeiT)
