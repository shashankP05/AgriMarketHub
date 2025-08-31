# Coconut Classification Project

This project uses YOLOv5 deep learning to detect and classify coconuts into three categories: dry coconut, mature coconut, and tender coconut.

## Files

### Training and Evaluation
- `Coconut1.ipynb`: Jupyter notebook for training and evaluating the YOLOv5 coconut classification model in Google Colab

### Inference
- `Coconut.py`: Python script for running detection on individual images using the trained model

## What Happens in Each File

### `Coconut1.ipynb`
This Google Colab notebook handles the complete training pipeline:

1. **Environment Setup**: Installs required packages (ultralytics, roboflow, kaggle)
2. **YOLOv5 Repository**: Clones and sets up the official YOLOv5 repository
3. **Dataset Download**: Downloads coconut dataset from Kaggle using API
4. **Data Processing**: Organizes images and labels into YOLO format with train/val splits
5. **YAML Configuration**: Creates dataset configuration file with 3 classes
6. **Model Training**: Trains YOLOv5s model for 100 epochs with data augmentation
7. **Validation**: Evaluates model performance on validation set
8. **Testing**: Runs inference on sample images to verify detection
9. **Results Visualization**: Displays training plots, confusion matrix, and sample predictions
10. **Custom Inference**: Loads trained model and tests on custom images
11. **Download Results**: Packages and downloads all training results, models, and sample predictions

### `Coconut.py`
This inference script:
- Loads the trained YOLOv8 model weights (`best.pt`)
- Prompts user to enter an image path
- Runs coconut detection and classification
- Displays the image with bounding boxes and class labels
- Prints detected coconut types with confidence scores

## Requirements

```
torch>=1.8.0
ultralytics
opencv-python
pillow
pyyaml
requests
scipy
matplotlib
pathlib
```

## Installation

Install the required packages:
```bash
pip install ultralytics torch torchvision opencv-python pillow
```

## Usage

### Training (Google Colab)
1. Open `Coconut1.ipynb` in Google Colab
2. Upload your `kaggle.json` API key when prompted
3. Run all cells sequentially
4. The notebook will automatically download the dataset, train the model, and provide results
5. Download the results zip file containing trained weights and evaluation metrics

### Detection (Local)
1. Download your trained model weights (`best.pt`) from the Colab training results
2. Update the `MODEL_PATH` variable in `Coconut.py` to point to your model weights:
   ```python
   MODEL_PATH = r"path/to/your/best.pt"
   ```
3. Run the detection script:
   ```bash
   python Coconut.py
   ```
4. When prompted, enter the full path to the image you want to analyze
5. The script will display the image with detected coconuts and print classification results

## Model Classes

The model can detect and classify three types of coconuts:
- **dry_coconut**: Mature coconuts that have been dried
- **mature_coconut**: Fully grown coconuts ready for harvest
- **tender_coconut**: Young, green coconuts with tender flesh

## Technologies Used

- **YOLOv5**: Object detection and classification framework
- **PyTorch**: Deep learning framework
- **Google Colab**: Cloud-based training environment
- **Kaggle API**: Dataset downloading
- **OpenCV**: Image processing
- **Ultralytics**: YOLOv5 implementation and utilities

## Training Details

- **Model**: YOLOv5s (small variant for balance of speed and accuracy)
- **Image Size**: 640x640 pixels
- **Batch Size**: 16
- **Epochs**: 100 (with early stopping patience of 20)
- **Data Augmentation**: Automatic augmentations included
- **Validation Split**: Automated train/validation split from dataset

## Troubleshooting

1. **Kaggle API Error**: Ensure your `kaggle.json` file is properly uploaded and contains valid credentials
2. **CUDA Memory Error**: Reduce batch size in training configuration
3. **Path Errors**: Use raw strings (r"path") or forward slashes for file paths on Windows
4. **Model Loading Error**: Ensure the `MODEL_PATH` points to the correct location of your `best.pt` file

## Results

After training, you'll receive:
- Trained model weights (`best.pt` and `last.pt`)
- Training plots showing loss and accuracy curves
- Confusion matrix for classification performance
- Sample detection images
- Detailed