import cv2
import numpy as np
from ultralytics import YOLO


def get_corrected_bbox(image_path, model, min_size=50):
    """
    Args: image_path (str) - path to image
          model (YOLO)     - trained YOLO model for pallet detection
          min_size (float) - minimum pixel dimension to be considered valid
    Returns: min_w (float) - corrected pixel width accounting for rotation
             min_h (float) - corrected pixel height accounting for rotation
    """
    import os
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    min_area = float('inf')
    best_w = None
    best_h = None
    best_angle = 0
    best_rotated = None
    best_result = None

    for angle in range(0, 180, 1):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        temp_path = f'temp_rotated_{angle}.png'
        cv2.imwrite(temp_path, rotated)

        results = model(temp_path, verbose=False)

        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls)] == 'pallet':
                    xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()

                    if pixel_w < min_size or pixel_h < min_size:
                        continue

                    aspect_ratio = max(pixel_w, pixel_h) / min(pixel_w, pixel_h)
                    if aspect_ratio > 2.0:
                        continue

                    area = pixel_w * pixel_h
                    if area < min_area:
                        min_area = area
                        best_w = pixel_w
                        best_h = pixel_h
                        best_angle = angle
                        best_rotated = rotated.copy()
                        best_result = result

        os.remove(temp_path)

    if best_w is None:
        return float('inf'), float('inf')

    # Save annotated best angle image
    if best_result is not None and best_rotated is not None:
        annotated = best_result.plot()
        save_name = f"measurement_{os.path.splitext(os.path.basename(image_path))[0]}_angle{best_angle}.png"
        cv2.imwrite(save_name, annotated)
        print(f"Saved measurement image: {save_name}")

    print(f"Best angle:        {best_angle} degrees")
    print(f"Corrected bbox:    {best_w:.1f} x {best_h:.1f} px")

    return best_w, best_h

def measure_pallet(image_path, model, real_w_inches=40, real_h_inches=48):
    """
    Args:    image_path (str)      - path to reference image with known pallet size
             model (YOLO)          - trained YOLO model for pallet detection
             real_w_inches (float) - known real world width in inches
             real_h_inches (float) - known real world height in inches
    Returns: scale_w (float)       - inches per pixel width
             scale_h (float)       - inches per pixel height

    Calibrates pixel to inch scale factors using a reference image with known
    pallet dimensions, correcting for rotation via bbox minimization.
    """
    pixel_w, pixel_h = get_corrected_bbox(image_path, model)

    if pixel_w == float('inf'):
        print("No pallet detected in reference image")
        return None, None

    scale_w = real_w_inches / pixel_w
    scale_h = real_h_inches / pixel_h

    print(f"Scale factors:     {scale_w:.4f} in/px (w) | {scale_h:.4f} in/px (h)")
    print(f"Real dimensions:   {pixel_w * scale_w:.1f} x {pixel_h * scale_h:.1f} inches")

    return scale_w, scale_h


def estimate_pallet_size(image_path, model, scale_w, scale_h):
    """
    Args:    image_path (str) - path to image with unknown pallet size
             model (YOLO)     - trained YOLO model for pallet detection
             scale_w (float)  - inches per pixel width from calibration
             scale_h (float)  - inches per pixel height from calibration
    Returns: real_w (float)   - estimated real world width in inches
             real_h (float)   - estimated real world height in inches

    Estimates real world pallet dimensions using calibrated scale factors,
    correcting for rotation via bbox minimization.
    """
    pixel_w, pixel_h = get_corrected_bbox(image_path, model)

    if pixel_w == float('inf'):
        print("No pallet detected in image")
        return None, None

    real_w = pixel_w * scale_w
    real_h = pixel_h * scale_h

    print(f"Estimated size:    {real_w:.1f} x {real_h:.1f} inches")

    return real_w, real_h


if __name__ == '__main__':
    model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')

    # Step 1 — Calibrate from reference image
    scale_w, scale_h = measure_pallet('image_to_measure.png', model)

    # Step 2 — Estimate unknown pallet
    if scale_w and scale_h:
        estimate_pallet_size('unknown_pallet.png', model, scale_w, scale_h)
        estimate_pallet_size('test_image_measure.png', model, scale_w, scale_h)