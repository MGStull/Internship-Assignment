from ultralytics import YOLO

model = YOLO(r'runs\detect\intership-assignment\pallet-detection\weights\best.pt')
results = model('image_to_measure.png')

for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        
        if class_name == 'pallet':
            xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()
            
            # Known real world dimensions
            real_w_inches = 40  # inches
            real_h_inches = 48  # inches
            
            # Pixel to inch conversion
            scale_w = real_w_inches / pixel_w
            scale_h = real_h_inches / pixel_h
            
            print(f"Pixel dimensions:  {pixel_w:.1f} x {pixel_h:.1f} px")
            print(f"Scale factors:     {scale_w:.7f} in/px (w)  |  {scale_h:.7f} in/px (h)")
            print(f"Real dimensions:   {pixel_w * scale_w:.1f} x {pixel_h * scale_h:.1f} inches")


import numpy as np

measurements_w = []
measurements_h = []
runs = 10
"""
# Testing for possible Jitter in measurements by running multiple times and calculating error range
for _ in range(runs):
    unknown_results = model('unknown_pallet.png')
    for result in unknown_results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'pallet':
                xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()
                measurements_w.append(pixel_w * scale_w)
                measurements_h.append(pixel_h * scale_h)

# Step 3 — Calculate error range
mean_w = np.mean(measurements_w)
mean_h = np.mean(measurements_h)
std_w  = np.std(measurements_w)
std_h  = np.std(measurements_h)

print(f"Estimated Width:  {mean_w:.1f} ± {std_w:.2f} inches")
print(f"Estimated Height: {mean_h:.1f} ± {std_h:.2f} inches")
print(f"Width  range:     {mean_w - 2*std_w:.1f} — {mean_w + 2*std_w:.1f} inches")
print(f"Height range:     {mean_h - 2*std_h:.1f} — {mean_h + 2*std_h:.1f} inches")
"""
print("\n Error Range Skipped, more examples would be needed for accurate error estimation. See Decisions.md for details.")

##Just a random example I grabbed to demonstrate the method


unknown_results = model('unknown_pallet.png')
for result in unknown_results:
    for box in result.boxes:
        if model.names[int(box.cls)] == 'pallet':
            xc, yc, pixel_w, pixel_h = box.xywh[0].tolist()
            
            real_w = pixel_w * scale_w
            real_h = pixel_h * scale_h

            print(f"Estimated size: {real_w:.1f} x {real_h:.1f} inches")


def measure_pallet(image_path, model, real_w_inches=40, real_h_inches=48):
    """
    Measures pallet dimensions in real world units using a reference image.
    
    Args:
        image_path:      Path to the reference image with known dimensions
        model:           Loaded YOLO model
        real_w_inches:   Known real world width in inches (default 40)
        real_h_inches:   Known real world height in inches (default 48)
    
    Returns:
        scale_w:         Inches per pixel (width)
        scale_h:         Inches per pixel (height)
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
    Estimates real world pallet dimensions using precomputed scale factors.
    
    Args:
        image_path:  Path to the unknown pallet image
        model:       Loaded YOLO model
        scale_w:     Inches per pixel (width) from measure_pallet()
        scale_h:     Inches per pixel (height) from measure_pallet()
    
    Returns:
        real_w:      Estimated width in inches
        real_h:      Estimated height in inches
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