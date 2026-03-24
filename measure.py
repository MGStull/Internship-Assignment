from ultralytics import YOLO

model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')
results = model('image_to_measure.png')

def measure_pallet(image_path, model, real_w_inches=40, real_h_inches=48):
    """
    args : image_path (str) - path to reference image with known pallet size
              model (YOLO) - trained YOLO model for pallet detection
              real_w_inches (float) - known real world width in inches
              real_h_inches (float) - known real world height in inches
    returns: scale_w (float) - inches per pixel width
             scale_h (float) - inches per pixel height

    This function is made to measure the pixel height and then create a conversion factor to real world inches using
    a baseline image with a measured pallet.
    """
    
    results = model(image_path)
    
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'pallet':
                xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()
                
                scale_w = real_w_inches / pixel_w
                scale_h = real_h_inches / pixel_h
                
                print(f"Pixel dimensions:  {pixel_w:.1f} x {pixel_h:.1f} px")
                print(f"Scale factors:     {scale_w:.4f} in/px (w) | {scale_h:.4f} in/px (h)")
                print(f"Real dimensions:   {pixel_w * scale_w:.1f} x {pixel_h * scale_h:.1f} inches")
                
                return scale_w, scale_h
    
    print("No pallet detected in reference image")
    return None, None


def estimate_pallet_size(image_path, model, scale_w, scale_h):
    """
    Args: image_path (str) - path to image with unknown pallet size
          model (YOLO) - trained YOLO model for pallet detection
          scale_w (float) - inches per pixel width from calibration
          scale_h (float) - inches per pixel height from calibration
    Returns: real_w (float) - estimated real world width in inches
             real_h (float) - estimated real world height in inches
    This function uses the baseline scale factors established to estimate the size of a pallet.

    """
    results = model(image_path)
    
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'pallet':
                xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()
                
                real_w = pixel_w * scale_w
                real_h = pixel_h * scale_h
                
                print(f"Pixel dimensions:  {pixel_w:.1f} x {pixel_h:.1f} px")
                print(f"Estimated size:    {real_w:.1f} x {real_h:.1f} inches")
                
                return real_w, real_h
    
    print("No pallet detected in image")
    return None, None


# Usage
if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')

    # Step 1 — Calibrate from reference image
    scale_w, scale_h = measure_pallet('image_to_measure.png', model)

    # Step 2 — Estimate unknown pallet
    if scale_w and scale_h:
        estimate_pallet_size('unknown_pallet.png', model, scale_w, scale_h)