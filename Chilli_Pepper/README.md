# Chili Ripeness Detection - Documentation

## Overview
This project uses a deep learning model (YOLOv8, via the Ultralytics Python package) to detect and classify the ripeness of chilies in images. The model can identify classes such as "ripe", "unripe", and "overripe" based on your training data.

## Requirements
- Python 3.8–3.12 (recommended: Python 3.12.8)
- NVIDIA GPU (optional, for faster inference)
- The following Python packages:
  - ultralytics
  - torch
  - opencv-python
  - matplotlib
  - numpy
  - pillow
  - pyyaml
  - requests
  - scipy

Install all requirements with:
```
pip install ultralytics torch opencv-python matplotlib numpy pillow pyyaml requests scipy
```

## Files
- `detect.py`: Script to run inference on a single image using a trained YOLOv8 model.
- `chilleripedetect.ipynb`: Jupyter notebook for training and evaluating the model.

## How to Use
1. After training in the notebook, download your trained model weights (e.g., `best.pt`).
2. In `detect.py`, set the path to your model weights (`best.pt`) in the script.
3. Run the detection script:
   ```
   python detect.py
   ```
4. When prompted, enter the path to the image you want to analyze.
5. The script will display the image with detected chilies and print the ripeness class and confidence for each detection.

## What We Used
- **YOLOv8** (Ultralytics): State-of-the-art object detection framework.
- **PyTorch**: Deep learning library for model training and inference.
- **OpenCV**: For image processing and visualization.
- **Jupyter Notebook**: For training and experimentation.

## What Happens in Each File

### chilleripedetect.ipynb
This Jupyter notebook is used for training and evaluating a deep learning model to detect and classify chili ripeness. It uses the Ultralytics YOLOv8 framework. Typical steps include:
- Loading and preparing a dataset of chili images.
- Training a YOLOv8 model to recognize different ripeness stages (e.g., ripe, unripe, overripe).
- Evaluating the model’s performance and saving the trained weights (usually as `best.pt`).

### detect.py
This Python script runs inference (detection) using the trained YOLOv8 model. When you run the script:
1. It prompts you to enter the path to an image.
2. Loads the trained YOLOv8 model using the Ultralytics API.
3. Runs detection on the provided image.
4. Displays the image with bounding boxes and class labels (e.g., “ripe”, “unripe”).
5. Prints the detected class labels and confidence scores to the console.

## Notes
- Make sure your model was trained with YOLOv8 for compatibility.
- For batch or advanced inference, refer to the Ultralytics documentation: https://docs.ultralytics.com/

## Troubleshooting
- If you see errors about model compatibility, ensure you are using YOLOv8 and the latest `ultralytics` package.
- If you have issues with Python versions, use Python 3.12.x for best results.

## References
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/)
- [OpenCV Documentation](https://opencv.org/)
