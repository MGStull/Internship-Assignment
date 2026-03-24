import os
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch


##Problem 1
def __main__():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8s.pt').to(device)  # Load model to GPU

    results = model.train(
        data='dataset/dataset/data.yaml',
        epochs=100,
        imgsz=320,
        batch=32,        
        device=0,        
        project='intership-assignment',
        name='pallet-detection'
    )


if __name__ == "__main__":
    __main__()