import os
from collections import Counter

print("Class distribution in train set:")
label_dir = "dataset/dataset/train/labels"  # adjust path as needed
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} annotations")

print("\nClass distribution in val set:")
label_dir = "dataset/dataset/val/labels"  # adjust path as needed
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} annotations")

print("\nTest set annotation counts:")
label_dir = "dataset/dataset/test/labels"  # adjust path as needed
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} annotations")


