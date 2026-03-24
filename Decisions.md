
## .gitignore
**Date** 3/23/26
**Decision** Datasets added to .gitignore
**Reasoning** To ensure data privacy and prevent data leaks, also deleted __MACOSX since I run windows

## For the training of the first model I ellected to use YOLOv8 Object detection models.
**Date:** 3/23/26  
**Decision:**  v8 Object detection 
**Reasoning**  Stanadard and pragmatic choice for task.

## Class set size reduced
**Date:** 3/23/26
**Decision** From 16 to 6 based on class distribution may consider reducing to 5 since class 14 has only 8 annotations
**Reasoning**
Class Organization
    Class 0: 161 annotations
    Class 1: 818 annotations
    Class 2: 478 annotations
    Class 4: 292 annotations
    Class 6: 46 annotations
    Class 14: 8 annotations
## WARNING
**Date** 3/23/26
**Decision** Model may be unstable fro Class 14 due to data limitation, but has stabalized on other classes
**Reasoning** Data Limitation on Class14 poses a instability risk.

## Warning
**date** 3/23/26
**Decisions** Will not include classes 6 and 14 in test results
**Reason** No test images available in test set :
    TEST SET CLASS ALLOCATION
    Class 0: 10 annotations
    Class 1: 59 annotations
    Class 2: 31 annotations
    Class 4: 20 annotations

## Finished Problem 1 with good results
**Date** 3/23/26
**Decision** Model is adequatly trained on given data for Problem 1
**Reasoning** Test set Results indicate strong gerneralization for given 
    Test Set Metrics
    mAP@50:        0.9887
    mAP@50-95:     0.8470
    Precision:     0.9641
    Recall:        0.9828

    Per Class Metrics
    pallet               AP@50: 0.9950
    board                AP@50: 0.9727
    stringer             AP@50: 0.9922
    stringerProfile      AP@50: 0.9950


## Error Range Calculation
**Date** 3/23/26
**Decision** Remove Error Range Calculation
**Reasoning** YOLOv8 is deterministic error range only comes out to 0, for proper error range, multiple examples are needed.

## Color Classification
**Date** 3/23/26
**Decision** Use kmeans clustering with vectorization of pixels for identification of pallets as a certain color
**Reasoning** These hsv values are easy to find online for pallet/wood clustering based on color, so it will be quick to implement and is great for low data enviornments
It also allows for the provided images of pallets to be used to test the clustering

## Model/Data Storage
**Date** 3/23/26
**Decision** Moving models back into default loactions so that access it runs without needing manual changes, also moved dataset to be in default location when xip-extracted into repository loaction '/Internship-Assignment'
**Reasoning** In case Amir or Guy would like to run the model train or any file without having to make the edits I made to file locations it should be more plug and play for evaluation and running should be standardized. Also helps to show that the results are repeatable. Also to preserve the validity of results the original success will be included since the test set was run on the original model. I will also include the new Test Results for the second run below.

Train 
Model summary (fused): 73 layers, 11,131,776 parameters, 0 gradients, 28.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.9it/s 0.3s
                   all         30        338      0.877      0.991      0.985      0.757
                pallet         30         30      0.976          1      0.995      0.992
                 board         30        162      0.957      0.981      0.993      0.855
              stringer         30         88      0.902      0.989      0.947      0.671
             leadBoard         30         57      0.934      0.982      0.994      0.869
           brokenBoard          1          1      0.616          1      0.995      0.398

Ultralytics 8.4.19  Python-3.13.12 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3090, 24576MiB)
Model summary (fused): 73 layers, 11,131,776 parameters, 0 gradients, 28.5 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1099.6110.4 MB/s, size: 204.0 KB)
val: Scanning C:\Users\chick\Documents\Internship-Assignment\dataset\dataset\test\labels.cache... 10 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 10/10 1.6Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 6.8s/it 6.8s
                   all         10        120      0.964      0.983      0.989      0.847
                pallet         10         10      0.939          1      0.995      0.995
                 board         10         59      0.982      0.931      0.973      0.794
              stringer         10         31      0.964          1      0.992      0.745
             leadBoard         10         20      0.971          1      0.995      0.855
Speed: 1.4ms preprocess, 23.1ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to C:\Users\chick\Documents\Internship-Assignment\runs\detect\zira-assessment\test-results2

 Test Set Metrics
mAP@50:        0.9887
mAP@50-95:     0.8470
Precision:     0.9641
Recall:        0.9828

 Per Class Metrics
pallet               AP@50: 0.9950
board                AP@50: 0.9727
stringer             AP@50: 0.9922
stringerProfile      AP@50: 0.9950