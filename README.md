# Cataract AI Diagnostic System

A deep learning-based application for detecting cataracts using both **Fundus** and **Slit-Lamp** eye images.

## Features
- **Dual-Model Architecture**:
  - **Fundus Analysis**: Binary classification (Normal vs. Cataract).
  - **Slit-Lamp Analysis**: Multi-class classification (Normal, Immature, Mature).
- **Preprocessing Pipeline**: 
  - Green Channel Extraction
  - Denoising (Gaussian Blur)
  - Contrast Enhancement (CLAHE)
- **Interactive Web UI**: Built with Flask, featuring drag-and-drop uploads and visualization of preprocessing steps.
- **Deep Learning**: Powered by **DenseNet169**.

## Directory Structure
```
├── Cataract/              # Raw Fundus images
├── slit-lamp/             # Raw Slit-Lamp images (normal, immature, mature)
├── models/                # Model definitions (densenet.py)
├── preprocessing/         # Image processing pipeline
├── saved_models/          # Trained model weights (.pth)
├── training/              # Training scripts
│   ├── train.py           # Fundus model training
│   └── train_slit_lamp.py # Slit-Lamp model training
├── ui/                    # Flask Web Application
│   ├── app.py             # Backend logic
│   ├── static/            # CSS, JS, Uploads
│   └── templates/         # HTML templates
└── config.py              # Global configuration
```

## Installation

1. **Clone the repository** (if applicable).
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy opencv-python Flask Pillow tqdm
   ```
   *Note: Ensure you have a CUDA-enabled GPU for faster training, though CPU is supported.*

## Usage

### 1. Training the Models
The application requires trained models to function effectively.

**Train Fundus Model:**
```bash
python training/train.py --epochs 20
```

**Train Slit-Lamp Model:**
```bash
python training/train_slit_lamp.py --epochs 20
```
*Models are saved to the `saved_models/` directory.*

### 2. Running the Application
Start the Flask web server:
```bash
python ui/app.py
```
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## How it Works
1. **Upload**: Select a Fundus or Slit-Lamp image in the respective section.
2. **Analysis**: The image goes through the preprocessing pipeline (Resize -> Green Channel -> Denoise -> CLAHE).
3. **Inference**: The processed image is passed to the specific DenseNet169 model.
4. **Result**: The UI displays the original image, preprocessing steps, diagnosis, and confidence score.

## Configuration
Adjust hyperparameters and paths in `config.py`:
- `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`
- `IMAGE_SIZE` (default 224x224)
- Data directory paths
