from ultralytics import YOLO
from pathlib import Path

# Set your trained model weights path here
MODEL_PATH = r"C:\Users\shash\Desktop\Agrihub\Coconut\best.pt"

def main():
    image_path = input("Enter the path to the image: ").strip()
    if not Path(image_path).is_file():
        print("Invalid image path.")
        return

    print("Loading YOLOv8 coconut classification model...")
    
    # Load YOLOv8 model trained for coconut classification
    model = YOLO(MODEL_PATH)
    
    print("Running detection...")
    results = model(image_path)
    results[0].show()  # Display image with detections

    # Print detected class labels and confidence scores
    detections = results[0].boxes
    
    if detections is not None and len(detections) > 0:
        print(f"\nFound {len(detections)} detection(s):")
        for i, detection in enumerate(detections):
            confidence = detection.conf.item()
            class_id = int(detection.cls.item())
            class_name = model.names[class_id]
            print(f"Detection {i+1}: {class_name} (Confidence: {confidence:.3f})")
    else:
        print("\nNo coconuts detected in the image")

if __name__ == "__main__":
    main()