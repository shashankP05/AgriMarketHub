from ultralytics import YOLO

# Set your trained model path here
MODEL_PATH = "chili_training_results/weights/best.pt"

def main():
    image_path = input("Enter the path to the image: ").strip()
    model = YOLO(MODEL_PATH)
    results = model(image_path)
    results[0].show()  # Show image with detections

    # Print detected class labels and confidence scores
    detections = results[0].boxes
    if detections is not None and len(detections) > 0:
        print(f"Found {len(detections)} detection(s):")
        for i, detection in enumerate(detections):
            confidence = detection.conf.item()
            class_id = int(detection.cls.item())
            class_name = model.names[class_id]
            print(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f})")
    else:
        print("No chili detected in the image")

if __name__ == "__main__":
    main()