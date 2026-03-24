from ultralytics import YOLO
import cv2
from pathlib import Path

if __name__ == '__main__':
    
    # Load trained model
    model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')

    # Run on test set
    metrics = model.val(
        data='dataset/dataset/data.yaml',
        split='test',
        imgsz=320,
        device=0,
        project='zira-assessment',
        name='test-results'
    )

    # Print metrics
    print("\n Test Set Metrics ")
    print(f"mAP@50:        {metrics.box.map50:.4f}")
    print(f"mAP@50-95:     {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    print("\n Per Class Metrics")
    for i, ap in enumerate(metrics.box.ap50):
        class_name = model.names[i]
        print(f"{class_name:<20} AP@50: {ap:.4f}")