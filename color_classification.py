import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

def classify_pallet_color(image_path, model):
    
    results = model(image_path)
    image = cv2.imread(image_path)
    
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'pallet':
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                pallet_crop = image[y1:y2, x1:x2]
                
                hsv = cv2.cvtColor(pallet_crop, cv2.COLOR_BGR2HSV)
                
                pixels = hsv.reshape(-1, 3).astype(np.float32)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                dominant = kmeans.cluster_centers_[np.argmax(
                    np.bincount(kmeans.labels_)
                )]
                
                h, s, v = dominant
                
                color = classify_hsv(h, s, v)
                print(f"Pallet color: {color}")
                print(f"HSV values: H={h:.1f}, S={s:.1f}, V={v:.1f}")

def classify_hsv(h, s, v):
    # White wood — high value, low saturation
    if v > 180 and s < 40:
        return 'white'
    
    # Grey wood — mid value, low saturation
    elif v > 100 and s < 40:
        return 'grey'
    
    # Black — very low value
    elif v < 50:
        return 'black'
    
    # Red pallet — wraps around HSV spectrum (0-10 and 170-180)
    elif (h < 10 or h > 170) and s > 80:
        return 'red'
    
    # Yellow pallet
    elif 20 < h < 35 and s > 80:
        return 'yellow'
    
    # Green pallet
    elif 35 < h < 85 and s > 50:
        return 'green'
    
    # Blue pallet
    elif 90 < h < 130 and s > 50:
        return 'blue'
    
    # Natural/brown wood
    elif 10 < h < 30 and s > 40:
        return 'natural'
    
    else:
        return 'unknown'


images = [
    'images to Classify Color/classify_color_1.png',
    'images to Classify Color/classify_color_2.png',
    'images to Classify Color/classify_color_3.png',
    'images to Classify Color/classify_color_4.png',
    'unknown_pallet.png'
]
model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')
for img_path in images:
    classify_pallet_color(img_path, model)