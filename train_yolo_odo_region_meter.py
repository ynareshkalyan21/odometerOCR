# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
import os
import json
import shutil
import cv2
import random

import yaml
from ultralytics import YOLO
from config import base_dataset_path

yolo_odo_region_dataset_path = "_yolo_odo_region_dataset"

CLASS_MAPPING = {"LCD": 0, "odometer": 1}


# Function to convert polygon points to bounding box
def polygon_to_bbox(all_points_x, all_points_y):
    x_min, x_max = min(all_points_x), max(all_points_x)
    y_min, y_max = min(all_points_y), max(all_points_y)
    return x_min, y_min, x_max, y_max


# Function to normalize bounding box coordinates
def normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


# Function to check if one bounding box is inside another
def is_inside(inner_box, outer_box):
    x1_in, y1_in, x2_in, y2_in = inner_box
    x1_out, y1_out, x2_out, y2_out = outer_box
    return x1_in >= x1_out and y1_in >= y1_out and x2_in <= x2_out and y2_in <= y2_out


def create_data_yaml(output_path):
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "LCD",
            1: "odometer"
        }
    }
    yaml_output_path = os.path.join(output_path, "data.yaml")

    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)
    with open(yaml_output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


# Main function to process VIA JSON files and convert to YOLO format
def process_via_dataset(base_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Delete existing output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    images_output = os.path.join(output_path, "images")
    labels_output = os.path.join(output_path, "labels")

    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    # Automatically detect all subfolders (groups) in the base path
    groups = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    valid_images = 0
    invalid_images = 0
    missing_images = 0
    total_images = 0

    all_image_paths = []

    # Collect all image paths first
    for group in groups:
        group_path = os.path.join(base_path, group)
        json_path = os.path.join(group_path, "via_region_data.json")

        # Load VIA JSON file
        if not os.path.exists(json_path):
            print(f"No JSON found in {group_path}. Skipping.")
            continue

        with open(json_path, "r") as file:
            via_data = json.load(file)

        for file_key, data in via_data.items():
            filename = data["filename"]
            image_path = os.path.join(group_path, filename)
            # Check if image exists
            if os.path.exists(image_path):
                all_image_paths.append((image_path, data))
            else:
                missing_images += 1

    # Shuffle all images before splitting into train, val, and test sets
    random.shuffle(all_image_paths)

    # Calculate the split sizes
    total_files = len(all_image_paths)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size

    # Split dataset into train, val, and test
    splits = {
        'train': all_image_paths[:train_size],
        'val': all_image_paths[train_size:train_size + val_size],
        'test': all_image_paths[train_size + val_size:]
    }

    # Dataset insights per split
    dataset_insights = {
        'train': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
        'val': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
        'test': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0}
    }

    # Iterate over each split and process images
    for split_name, split_data in splits.items():
        split_images_output = os.path.join(images_output, split_name)
        split_labels_output = os.path.join(labels_output, split_name)
        os.makedirs(split_images_output, exist_ok=True)
        os.makedirs(split_labels_output, exist_ok=True)

        for image_path, data in split_data:
            filename = data["filename"]
            regions = data["regions"]
            img = cv2.imread(image_path)

            if img is None:
                dataset_insights[split_name]['missing'] += 1
                print(f"Failed to read {image_path}. Skipping.")
                continue

            img_height, img_width = img.shape[:2]

            # Copy image to the appropriate output folder
            new_image_path = os.path.join(split_images_output, filename)
            shutil.copy(image_path, new_image_path)

            # Initialize bounding boxes
            lcd_box = None
            odometer_box = None

            # Process regions
            yolo_annotations = []
            for region in regions:
                shape = region["shape_attributes"]
                identity = region["region_attributes"].get("identity", None)

                if identity not in CLASS_MAPPING:
                    continue  # Skip unknown classes

                # Extract polygon and convert to bbox
                all_points_x = shape["all_points_x"]
                all_points_y = shape["all_points_y"]
                bbox = polygon_to_bbox(all_points_x, all_points_y)

                # Store LCD and odometer bounding boxes
                if identity == "LCD":
                    lcd_box = bbox
                elif identity == "odometer":
                    odometer_box = bbox

                # Normalize coordinates for YOLO format
                normalized_bbox = normalize_bbox(*bbox, img_width, img_height)
                yolo_annotations.append(
                    f"{CLASS_MAPPING[identity]} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]}")

            # Check if odometer is inside LCD
            if lcd_box and odometer_box:
                if not is_inside(odometer_box, lcd_box):
                    dataset_insights[split_name]['invalid'] += 1
                    print(f"Odometer not inside LCD for image: {image_path}. Skipping.")
                    continue  # Skip this image if condition fails
                dataset_insights[split_name]['valid'] += 1

            # Save YOLO annotations to a text file
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(split_labels_output, label_filename)
            with open(label_path, "w") as label_file:
                label_file.write("\n".join(yolo_annotations))

            dataset_insights[split_name]['total'] += 1
            total_images += 1

    # Print dataset analysis
    print("\n=== Dataset Analysis ===")
    print(f"Total images processed: {total_images}")
    print(f"Missing images: {missing_images}")

    for split_name, insights in dataset_insights.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total images: {insights['total']}")
        print(f"  Valid images (odometer inside LCD): {insights['valid']}")
        print(f"  Invalid images (odometer outside LCD): {insights['invalid']}")
        print(f"  Missing images: {insights['missing']}")

    print(f"\nTotal valid images: {valid_images}")
    print(f"Total invalid images: {invalid_images}")
    print(f"Total missing images: {missing_images}")


def train_yolo_mode(output_dataset_path):
    # Load a pre-trained YOLO model (YOLOv8)
    model = YOLO("yolov8n.pt")  # 'yolov8n.pt' is a smaller, faster variant

    # Train the model
    model.train(data=f'{output_dataset_path}/data.yaml', epochs=50, imgsz=640, batch=16, project='yolo_model',
                name='odometer_lcd')

if __name__ == "__main__":
    process_via_dataset(base_dataset_path, yolo_odo_region_dataset_path)
    train_yolo_mode(yolo_odo_region_dataset_path)
