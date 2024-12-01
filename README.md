# CoDeiT: Contrastive Data-efficient Transformers for Deepfake Detection

[Link to Paper](https://drive.google.com/file/d/1B8xb3C8t4y0FgPLDhflEPJqReOCwYFXb/view)

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
git clone https://github.com/imjohnzakkam/CoDeiT.git
cd CoDeiT

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
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

## Acknowledgments

- Indian Institute of Information Technology Design and Manufacturing, Kancheepuram
- University of North Texas
- Shiv Nadar University
- The authors of FaceForensics++, Celeb-DF, and DFDC datasets

## Contact

John Zakkam - ced18i059@iiitdm.ac.in
or please create an issue / PR
